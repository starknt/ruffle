//! Run with an optimized native build. Inputs are supplied locally and never copied to results.
use super::mesh::{PreparedMesh, ShapeInput};
use ruffle_render::backend::RenderBackend;
use ruffle_render::{
    bitmap::{Bitmap, BitmapSize},
    shape_utils::DistilledShape,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{borrow::Cow, fs, hint::black_box, path::Path, sync::Arc, time::Instant};
use swf::Tag;

fn timed<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed().as_nanos() as u64)
}
struct Bench {
    device: wgpu::Device,
    queue: wgpu::Queue,
    file: tempfile::NamedTempFile,
}
impl Bench {
    fn compare<T>(
        &self,
        label: &str,
        generate: impl Fn() -> T,
        archive: impl Fn(&T) -> Vec<u8>,
        decode: impl Fn(Vec<u8>) -> T,
        upload: impl Fn(&T),
    ) {
        let source = generate();
        let (bytes, encode_ns) = timed(|| archive(&source));
        let digest = Sha256::digest(&bytes);
        let (_, write_ns) = timed(|| fs::write(self.file.path(), &bytes).unwrap());
        let load = || {
            let data = fs::read(self.file.path()).unwrap();
            assert_eq!(Sha256::digest(&data), digest);
            decode(data)
        };
        let restored = load();
        upload(&restored);
        let mut rows = Vec::new();
        for i in 0..7 {
            let sample = |use_file| {
                let (prepared, cpu) = timed(|| if use_file { load() } else { generate() });
                let (_, gpu) = timed(|| upload(black_box(&prepared)));
                [cpu, gpu]
            };
            let (regen, cache) = if i % 2 == 0 {
                (sample(false), sample(true))
            } else {
                let cache = sample(true);
                (sample(false), cache)
            };
            rows.push(json!({"regenerateNs":regen[0],"regeneratedUploadNs":regen[1],"loadVerifyDecodeNs":cache[0],"restoredUploadNs":cache[1]}));
        }
        println!(
            "DERIVED {}",
            json!({"case":label,"bytes":bytes.len(),"encodeNs":encode_ns,"writeNs":write_ns,"samples":rows})
        );
    }
    fn wait(&self) {
        let index = self.queue.submit([]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
    }
    fn upload_bitmap(&self, bitmap: &Bitmap<'_>) {
        let (w, h) = (bitmap.width(), bitmap.height());
        let rgba = bitmap.clone().to_rgba();
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let t = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            t.as_image_copy(),
            rgba.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: None,
            },
            size,
        );
        self.wait();
    }
    fn bitmap(&self, label: &str, generate: impl Fn() -> Bitmap<'static>) {
        let bitmap = generate();
        let (w, h) = (bitmap.width(), bitmap.height());
        let format = bitmap.format();
        let decode = |data: Vec<u8>| Bitmap::new(w, h, format, data);
        self.compare(
            label,
            generate,
            |b| b.data().to_vec(),
            decode,
            |bitmap| self.upload_bitmap(bitmap),
        );
        if let Ok(dest) = std::env::var("RUFFLE_BENCH_INPUT_RGBA")
            && (32..=1024).contains(&w)
            && (32..=1024).contains(&h)
            && !Path::new(&dest).exists()
        {
            let mut data = Vec::from(w.to_le_bytes());
            data.extend(h.to_le_bytes());
            data.extend(bitmap.to_rgba().data());
            fs::write(dest, data).unwrap();
        }
    }
    fn shader(&self, label: &str, generate: impl Fn() -> naga::Module) {
        self.compare(
            label,
            generate,
            |m| {
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(m)
                .unwrap();
                naga::back::wgsl::write_string(
                    m,
                    &info,
                    naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
                )
                .unwrap()
                .into_bytes()
            },
            |b| naga::front::wgsl::parse_str(std::str::from_utf8(&b).unwrap()).unwrap(),
            |m| {
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Naga(Cow::Owned(m.clone())),
                    });
                self.wait();
                black_box(shader);
            },
        );
    }
}
fn encoded<'a>(tag: &'a Tag<'a>) -> ruffle_render::bitmap::EncodedBitmap<'a> {
    use ruffle_render::bitmap::EncodedBitmap;
    match tag {
        Tag::DefineBitsJpeg2 { jpeg_data, .. } => EncodedBitmap::Jpeg {
            data: jpeg_data,
            alpha: None,
        },
        Tag::DefineBitsJpeg3(v) => EncodedBitmap::Jpeg {
            data: v.data,
            alpha: Some(v.alpha_data),
        },
        Tag::DefineBitsLossless(v) => EncodedBitmap::Lossless(v),
        _ => unreachable!(),
    }
}
fn benchmark_shape(
    backend: &mut crate::backend::WgpuRenderBackend<crate::target::TextureTarget>,
    input: ShapeInput,
    bitmaps: &crate::shape_recipe::FrozenBitmapSource,
    name: &str,
) {
    let cache = backend
        .descriptors
        .device
        .mesh_store()
        .expect("set RUFFLE_STATIC_MESH_CACHE=1")
        .clone();
    let mut cached = input.clone();
    cached.cache = Some(cache.clone());
    let budget = 32 << 20;
    let (first, first_ns) = timed(|| cached.clone().prepare_bounded(budget).unwrap());
    let length = first.retained_bytes();
    let owner = backend
        .descriptors
        .runtime
        .issue_owner(crate::gpu_runtime::OwnerKind::Graphic)
        .unwrap();
    let mut upload = |prepared: PreparedMesh| {
        let mesh = backend
            .register_prepared_shape(prepared, input.geometry.id, bitmaps, owner)
            .unwrap();
        let sub = backend.descriptors.queue.submit([]).unwrap();
        backend
            .descriptors
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(sub),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        black_box(mesh);
    };
    upload(first);
    benchmark_restores(
        name,
        length,
        first_ns,
        || cache.stats().0,
        |restore| {
            let shape = if restore { &cached } else { &input };
            shape.clone().prepare_bounded(budget).unwrap()
        },
        upload,
    );
}
fn benchmark_restores<T>(
    name: &str,
    length: u64,
    first_ns: u64,
    hits: impl Fn() -> u64,
    prepare: impl Fn(bool) -> T,
    mut upload: impl FnMut(T),
) {
    let before = hits();
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(5));
        let _ = prepare(true);
        if hits() > before {
            break;
        }
    }
    assert!(hits() > before, "archive was not restored: {name}");
    let mut samples = Vec::new();
    for index in 0..7 {
        let mut sample = |restore| {
            let start = hits();
            let (value, cpu) = timed(|| prepare(restore));
            assert_eq!(hits() - start, u64::from(restore));
            let (_, gpu) = timed(|| upload(value));
            (cpu, gpu)
        };
        let (fresh, restored) = if index % 2 == 0 {
            (sample(false), sample(true))
        } else {
            let restored = sample(true);
            (sample(false), restored)
        };
        samples.push(json!({"regenerateNs":fresh.0,"regeneratedUploadNs":fresh.1,"loadVerifyDecodeNs":restored.0,"restoredUploadNs":restored.1}));
    }
    println!(
        "DERIVED {}",
        json!({"case":name,"bytes":length,"firstPrepareNs":first_ns,"samples":samples})
    );
}
// Make a one-frame display of a measured asset without copying game logic into acceptance runs.
fn write_shape_fixture(
    path: &Path,
    name: &str,
    shape: &swf::Shape,
    bitmaps: &[(u32, u16, &Tag<'_>)],
) {
    let bounds = shape.shape_bounds;
    let padding = swf::Twips::from_pixels_i32(4);
    let header = swf::Header {
        compression: swf::Compression::None,
        version: 10,
        stage_size: swf::Rectangle {
            x_min: swf::Twips::ZERO,
            y_min: swf::Twips::ZERO,
            x_max: bounds.width() + padding + padding,
            y_max: bounds.height() + padding + padding,
        },
        frame_rate: swf::Fixed8::from_f32(30.0),
        num_frames: 1,
    };
    let geo = ruffle_render::shape_utils::OwnedDistilledShape::from(DistilledShape::from(shape));
    let mut tags = Vec::new();
    let ids: std::collections::BTreeSet<_> = geo.bitmap_ids().collect();
    for (_, id, tag) in bitmaps {
        if ids.contains(id) {
            tags.push(match tag {
                Tag::DefineBitsJpeg2 { id, jpeg_data } => {
                    Tag::DefineBitsJpeg2 { id: *id, jpeg_data }
                }
                Tag::DefineBitsJpeg3(value) => Tag::DefineBitsJpeg3(value.clone()),
                Tag::DefineBitsLossless(value) => Tag::DefineBitsLossless(value.clone()),
                _ => unreachable!(),
            });
        }
    }
    tags.push(Tag::DefineShape(Box::new(shape.clone())));
    tags.push(Tag::PlaceObject(Box::new(swf::PlaceObject {
        version: 2,
        action: swf::PlaceObjectAction::Place(shape.id),
        depth: 1,
        matrix: Some(swf::Matrix::translate(
            padding - bounds.x_min,
            padding - bounds.y_min,
        )),
        color_transform: None,
        ratio: None,
        name: None,
        clip_depth: None,
        class_name: None,
        filters: None,
        background_color: None,
        blend_mode: None,
        clip_actions: None,
        has_image: false,
        is_bitmap_cached: None,
        is_visible: None,
        amf_data: None,
    })));
    tags.push(Tag::ShowFrame);
    swf::write_swf(
        &header,
        &tags,
        fs::File::create(path.join(format!("{name}-{}.swf", shape.id))).unwrap(),
    )
    .unwrap();
}
fn walk<'a>(tags: &'a [Tag<'a>], all: &mut Vec<&'a Tag<'a>>) {
    for tag in tags {
        all.push(tag);
        if let Tag::DefineSprite(sprite) = tag {
            walk(&sprite.tags, all)
        }
    }
}
#[test]
#[ignore = "requires a native adapter and RUFFLE_BENCH_MOVIES (local SWFs separated by the platform path separator)"]
fn native_derived_costs() {
    let paths = std::env::var_os("RUFFLE_BENCH_MOVIES").expect("set RUFFLE_BENCH_MOVIES");
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = futures::executor::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .unwrap();
    let (device, queue) =
        futures::executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
    let bench = Bench {
        device,
        queue,
        file: tempfile::NamedTempFile::new().unwrap(),
    };
    let agal: &[u8] = include!("../../../naga-agal/tests/raytrace_fragment.agal");
    bench.shader("AGAL raytrace fragment", || {
        naga_agal::agal_to_naga(&naga_agal::parse_bytecode(agal).unwrap(), &[None; 8]).unwrap()
    });
    let pbj = include_bytes!(
        "../../../../tests/tests/swfs/avm2/blend_shader_luma_lighten/LumaLighten.pbj"
    );
    bench.shader("PBJ LumaLighten fragment", || {
        naga_pixelbender::ShaderBuilder::build(
            &ruffle_render::pixel_bender::parse_shader(pbj, true).unwrap(),
        )
        .unwrap()
        .fragment
    });
    let private = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
    let mut renderer = crate::backend::WgpuRenderBackend::for_offscreen(
        (64, 64),
        wgpu::Backends::VULKAN,
        wgpu::PowerPreference::HighPerformance,
    )
    .unwrap();
    // SAFETY: the test owns this newly created cache directory.
    unsafe {
        renderer
            .descriptors
            .device
            .enable_pipeline_cache(&private.path().join("cache"), "bench");
    }
    for path in std::env::split_paths(&paths) {
        let name = path.file_name().unwrap().to_string_lossy();
        let stream = swf::decompress_swf(fs::File::open(&path).unwrap()).unwrap();
        let movie = swf::parse_swf(&stream).unwrap();
        let mut tags = vec![];
        walk(&movie.tags, &mut tags);
        let mut shapes = vec![];
        let mut bitmaps = vec![];
        for tag in tags {
            match tag {
                Tag::DefineBitsJpeg2 { id, jpeg_data } => {
                    if let Ok((w, h)) =
                        ruffle_render::utils::decode_define_bits_jpeg_dimensions(jpeg_data)
                    {
                        bitmaps.push((w * h, *id, tag));
                    }
                }
                Tag::DefineBitsJpeg3(v) => {
                    if let Ok((w, h)) =
                        ruffle_render::utils::decode_define_bits_jpeg_dimensions(v.data)
                    {
                        bitmaps.push((w * h, v.id, tag));
                    }
                }
                Tag::DefineBitsLossless(v) => {
                    bitmaps.push((v.width as u32 * v.height as u32, v.id, tag));
                }
                Tag::DefineShape(s) => shapes.push(s.as_ref()),
                _ => {}
            }
        }
        println!(
            "INVENTORY {}",
            json!({"movie":name,"bitmaps":bitmaps.len(),"shapes":shapes.len()})
        );
        bitmaps.sort_by_key(|b| b.0);
        shapes.sort_by_key(|s| s.shape.len());
        if let Some(dir) = std::env::var_os("RUFFLE_BENCH_SCENE_DIR")
            && let Some(shape) = shapes.last()
        {
            write_shape_fixture(Path::new(&dir), &name, shape, &bitmaps);
        }
        let positions = |n: usize| -> Vec<usize> {
            if n == 0 {
                return vec![];
            };
            let mut out = vec![n / 2, n - 1];
            out.dedup();
            out
        };
        for i in positions(bitmaps.len()) {
            let (_, id, t) = bitmaps[i];
            let source = encoded(t);
            bench.bitmap(&format!("{name} bitmap {id}"), || source.decode().unwrap());
            let pixel = source.decode().unwrap();
            let length = pixel.data().len() as u64;
            if (128 << 10..=4 << 20).contains(&length) {
                let store = renderer
                    .descriptors
                    .device
                    .decoded_store()
                    .expect("set RUFFLE_BITMAP_DECODE_CACHE=1");
                let size = BitmapSize {
                    width: pixel.width(),
                    height: pixel.height(),
                };
                let (first, first_ns) = timed(|| store.decode(source, size).unwrap());
                bench.upload_bitmap(&first);
                benchmark_restores(
                    &format!("{name} complete bitmap {id}"),
                    length,
                    first_ns,
                    || store.stats().0,
                    |restore| {
                        if restore {
                            store.decode(source, size).unwrap()
                        } else {
                            source.decode().unwrap()
                        }
                    },
                    |image| bench.upload_bitmap(&image),
                );
            }
        }
        for i in positions(shapes.len()) {
            let s = shapes[i];
            let geometry = Arc::new(ruffle_render::shape_utils::OwnedDistilledShape::from(
                DistilledShape::from(s),
            ));
            if super::mesh::input_vertices(&geometry) < 256 {
                continue;
            }
            let mut source = crate::shape_recipe::FrozenBitmapSource::default();
            let mut bitmap_sizes = vec![];
            for id in geometry.bitmap_ids() {
                if source.0.contains_key(&id) {
                    continue;
                }
                if let Some((_, _, tag)) = bitmaps.iter().find(|(_, key, _)| *key == id) {
                    let image = encoded(tag).decode().unwrap();
                    let size = BitmapSize {
                        width: image.width(),
                        height: image.height(),
                    };
                    let handle = renderer.register_bitmap(image).unwrap();
                    source.0.insert(id, (size, Some(handle)));
                    bitmap_sizes.push((id, size));
                }
            }
            let input = ShapeInput {
                cache: None,
                geometry,
                scale: 1.,
                bitmap_sizes,
            };
            benchmark_shape(
                &mut renderer,
                input,
                &source,
                &format!("{name} complete shape {}", s.id),
            );
        }
    }
}
