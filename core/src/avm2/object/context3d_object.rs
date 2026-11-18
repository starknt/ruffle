//! Object representation for Context3D objects

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::object::TObject;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::value::Value;
use crate::avm2_stub_method;
use crate::bitmap::bitmap_data::BitmapRawData;
use crate::context::{RenderContext, UpdateContext};
use gc_arena::{Collect, Gc, GcWeak};
use naga_agal::AgalError;
use ruffle_common::utils::HasPrefixField;
use ruffle_render::backend::{
    BufferUsage, Context3D, Context3DBlendFactor, Context3DCommand, Context3DCompareMode,
    Context3DStencilAction, Context3DTextureFormat, Context3DTriangleFace,
    Context3DVertexBufferFormat, ProgramType, Texture,
};
use ruffle_render::commands::CommandHandler;
use ruffle_render::matrix::Matrix;
use std::cell::Cell;
use std::rc::Rc;
use swf::{Rectangle, Twips};

use super::program_3d_object::Program3DObject;
use super::texture_object::TextureObject;
use super::{ClassObject, IndexBuffer3DObject, Stage3DObject, VertexBuffer3DObject};

#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct Context3DObject<'gc>(pub Gc<'gc, Context3DData<'gc>>);

#[derive(Clone, Collect, Copy, Debug)]
#[collect(no_drop)]
pub struct Context3DObjectWeak<'gc>(pub GcWeak<'gc, Context3DData<'gc>>);

impl<'gc> Context3DObject<'gc> {
    pub fn from_context(
        context: &mut UpdateContext<'gc>,
        context3d: Box<dyn Context3D>,
        stage3d: Stage3DObject<'gc>,
    ) -> Self {
        let class = context.avm2.classes().context3d;

        Context3DObject(Gc::new(
            context.gc(),
            Context3DData {
                base: ScriptObjectData::new(class),
                render_context: Cell::new(Some(context3d)),
                stage3d,
            },
        ))
    }

    pub fn stage3d(self) -> Stage3DObject<'gc> {
        self.0.stage3d
    }

    pub fn with_context_3d<R>(self, f: impl FnOnce(&mut dyn Context3D) -> R) -> R {
        // Temporarily take ownership of the Context3D instance.
        let cell = &self.0.render_context;
        let mut guard = scopeguard::guard(cell.take(), |stolen| cell.set(stolen));
        f(guard
            .as_deref_mut()
            .expect("Context3D is missing or already in use"))
    }

    fn with_context_3d_result<R>(
        self,
        f: impl FnOnce(&mut dyn Context3D) -> Result<R, ruffle_render::error::RenderFailure>,
    ) -> Result<R, ruffle_render::error::RenderFailure> {
        self.with_context_3d(|ctx| {
            ctx.health().check()?;
            f(ctx).map_err(|error| ctx.health().record(error))
        })
    }

    fn with_context_3d_render_result<R>(
        self,
        f: impl FnOnce(&mut dyn Context3D) -> Result<R, ruffle_render::error::Error>,
    ) -> Result<R, ruffle_render::error::Error> {
        self.with_context_3d(|ctx| {
            ctx.health().check()?;
            let result = f(ctx).map_err(|error| match error {
                ruffle_render::error::Error::Render(failure) => ctx.health().record(failure).into(),
                other => other,
            })?;
            ctx.health().check()?;
            Ok(result)
        })
    }

    pub fn configure_back_buffer(
        self,
        width: u32,
        height: u32,
        anti_alias: u32,
        depth_and_stencil: bool,
        wants_best_resolution: bool,
        wants_best_resolution_on_browser_zoom: bool,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::ConfigureBackBuffer {
                width,
                height,
                anti_alias,
                depth_and_stencil,
                wants_best_resolution,
                wants_best_resolution_on_browser_zoom,
            })
        })?;

        Ok(())
    }

    pub fn create_index_buffer(
        self,
        num_indices: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, ruffle_render::error::RenderFailure> {
        let index_buffer = self.with_context_3d_result(|ctx| {
            ctx.create_index_buffer(BufferUsage::StaticDraw, num_indices)
        })?;

        Ok(Value::Object(IndexBuffer3DObject::from_handle(
            activation,
            self,
            index_buffer,
        )))
    }

    #[expect(clippy::too_many_arguments)]
    pub fn create_texture(
        self,
        width: u32,
        height: u32,
        format: Context3DTextureFormat,
        optimize_for_render_to_texture: bool,
        streaming_levels: u32,
        class: ClassObject<'gc>,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        check_texture_stub(activation, format);
        let texture = self.with_context_3d_render_result(|ctx| {
            ctx.create_texture(
                width,
                height,
                format,
                optimize_for_render_to_texture,
                streaming_levels,
            )
        })?;

        Ok(Value::Object(TextureObject::from_handle(
            activation, self, texture, format, class,
        )))
    }

    pub fn create_vertex_buffer(
        self,
        num_vertices: u32,
        data_32_per_vertex: u8,
        usage: BufferUsage,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, ruffle_render::error::RenderFailure> {
        let handle = self.with_context_3d_result(|ctx| {
            ctx.create_vertex_buffer(usage, num_vertices, data_32_per_vertex)
        })?;

        Ok(Value::Object(VertexBuffer3DObject::from_handle(
            activation,
            self,
            handle,
            data_32_per_vertex,
        )))
    }

    pub fn upload_vertex_buffer_data(
        self,
        buffer: VertexBuffer3DObject<'gc>,
        data: &[u8],
        start_vertex: usize,
        data32_per_vertex: u8,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::UploadToVertexBuffer {
                buffer: buffer.handle(),
                data,
                start_vertex,
                data32_per_vertex,
            })
        })?;

        Ok(())
    }

    pub fn upload_index_buffer_data(
        self,
        buffer: IndexBuffer3DObject<'gc>,
        data: &[u8],
        start_offset: usize,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        let mut handle = buffer.handle();
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::UploadToIndexBuffer {
                buffer: &mut *handle,
                data,
                start_offset,
            })
        })?;

        Ok(())
    }

    pub fn set_vertex_buffer_at(
        self,
        index: u32,
        buffer: Option<(VertexBuffer3DObject<'gc>, Context3DVertexBufferFormat)>,
        buffer_offset: u32,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetVertexBufferAt {
                index,
                buffer: buffer.map(|(b, format)| (b.handle(), format)),
                buffer_offset,
            })
        })?;

        Ok(())
    }

    pub fn create_program(self, activation: &mut Activation<'_, 'gc>) -> Value<'gc> {
        Value::Object(Program3DObject::from_context(activation, self))
    }

    pub fn upload_shaders(
        self,
        program: Program3DObject<'gc>,
        vertex_shader_agal: Vec<u8>,
        fragment_shader_agal: Vec<u8>,
    ) -> Result<(), AgalError> {
        self.with_context_3d(|ctx| {
            ctx.upload_shaders(
                program.shader_module_handle(),
                vertex_shader_agal,
                fragment_shader_agal,
            )
        })
    }

    pub fn set_program(
        self,
        program: Option<Program3DObject<'gc>>,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        let module = program.and_then(|p| p.shader_module_handle().borrow().clone());

        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetShaders { module })
        })?;

        Ok(())
    }

    pub fn draw_triangles(
        self,
        index_buffer: IndexBuffer3DObject<'gc>,
        first_index: u32,
        mut num_triangles: i32,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        if num_triangles == -1 {
            // FIXME - should we error if the number of indices isn't a multiple of 3?
            num_triangles = (index_buffer.count() / 3) as i32;
        }
        let handle = index_buffer.handle();

        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::DrawTriangles {
                index_buffer: &*handle,
                first_index: first_index as usize,
                num_triangles: num_triangles as isize,
            })
        })?;

        Ok(())
    }

    pub fn set_program_constants(
        self,
        program_type: ProgramType,
        first_register: u32,
        matrix_raw_data_column_major: &[[u8; 4]],
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetProgramConstants {
                program_type,
                first_register,
                matrix_raw_data_column_major,
            })
        })?;

        Ok(())
    }

    pub fn set_culling(
        self,
        face: Context3DTriangleFace,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetCulling { face })
        })?;

        Ok(())
    }

    pub fn set_blend_factors(
        self,
        source_factor: Context3DBlendFactor,
        destination_factor: Context3DBlendFactor,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetBlendFactors {
                source_factor,
                destination_factor,
            })
        })?;

        Ok(())
    }

    pub fn set_render_to_texture(
        self,
        texture: Rc<dyn Texture>,
        enable_depth_and_stencil: bool,
        anti_alias: u32,
        surface_selector: u32,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetRenderToTexture {
                texture,
                enable_depth_and_stencil,
                anti_alias,
                surface_selector,
            })
        })?;

        Ok(())
    }

    pub fn set_render_to_back_buffer(self) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetRenderToBackBuffer)
        })?;

        Ok(())
    }

    pub fn present(self) {
        self.with_context_3d(|ctx| ctx.present())
    }

    // Renders our finalized frame to the screen, as part of the Ruffle rendering process.
    pub fn render(self, context: &mut RenderContext<'_, 'gc>) {
        self.with_context_3d(|context3d| {
            if context3d.should_render() {
                let handle = context3d.bitmap_handle();

                // FIXME - The Stage3D should not be scaled
                // when resizing the player window.
                let mut transform = context.transform_stack.transform();
                transform.matrix *= Matrix::translate(
                    Twips::from_pixels(self.stage3d().x()),
                    Twips::from_pixels(self.stage3d().y()),
                );

                context.commands.render_stage3d(handle, transform);
            }
        });
    }

    #[expect(clippy::too_many_arguments)]
    pub fn set_clear(
        self,
        red: f64,
        green: f64,
        blue: f64,
        alpha: f64,
        depth: f64,
        stencil: u32,
        mask: u32,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::Clear {
                red,
                green,
                blue,
                alpha,
                depth,
                stencil,
                mask,
            })
        })?;

        Ok(())
    }

    pub(crate) fn copy_bitmapdata_to_texture(
        self,
        source: &BitmapRawData<'gc>,
        dest: Rc<dyn Texture>,
        layer: u32,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        // Note - Flash appears to allow a source that's larger than the destination.
        // Let's leave in this assertion to see if there any real SWFS relying on this
        // behavior.
        assert!(
            source.width() <= dest.width(),
            "Source width {:?} larger than dest width {:?}",
            source.width(),
            dest.width()
        );
        assert!(
            source.height() <= dest.height(),
            "Source height {:?} larger than dest height {:?}",
            source.height(),
            dest.height()
        );

        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::CopyBitmapToTexture {
                source: source.pixels_bgra(),
                source_is_bgra: true,
                source_width: source.width(),
                source_height: source.height(),
                dest,
                layer,
            })
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "jpegxr"), allow(unused))]
    pub(crate) fn copy_pixels_to_texture(
        self,
        source: Vec<u8>,
        dest: Rc<dyn Texture>,
        layer: u32,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::CopyBitmapToTexture {
                source: &source,
                source_is_bgra: false,
                source_width: dest.width(),
                source_height: dest.height(),
                dest,
                layer,
            })
        })?;

        Ok(())
    }

    pub(crate) fn set_texture_at(
        self,
        sampler: u32,
        texture: Option<Rc<dyn Texture>>,
        cube: bool,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetTextureAt {
                sampler,
                texture,
                cube,
            })
        })?;

        Ok(())
    }

    pub(crate) fn set_color_mask(
        self,
        red: bool,
        green: bool,
        blue: bool,
        alpha: bool,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetColorMask {
                red,
                green,
                blue,
                alpha,
            })
        })?;

        Ok(())
    }

    pub(crate) fn set_depth_test(
        self,
        depth_mask: bool,
        pass_compare_mode: Context3DCompareMode,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetDepthTest {
                depth_mask,
                pass_compare_mode,
            })
        })?;

        Ok(())
    }

    pub(crate) fn create_cube_texture(
        self,
        size: u32,
        format: Context3DTextureFormat,
        optimize_for_render_to_texture: bool,
        streaming_levels: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        check_texture_stub(activation, format);
        let texture = self.with_context_3d_render_result(|ctx| {
            ctx.create_cube_texture(
                size,
                format,
                optimize_for_render_to_texture,
                streaming_levels,
            )
        })?;

        let class = activation.avm2().classes().cubetexture;

        Ok(Value::Object(TextureObject::from_handle(
            activation, self, texture, format, class,
        )))
    }

    pub(crate) fn set_sampler_state_at(
        self,
        sampler: u32,
        wrap: ruffle_render::backend::Context3DWrapMode,
        filter: ruffle_render::backend::Context3DTextureFilter,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetSamplerStateAt {
                sampler,
                wrap,
                filter,
            })
        })?;

        Ok(())
    }

    pub(crate) fn set_scissor_rectangle(
        self,
        rect: Option<Rectangle<Twips>>,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetScissorRectangle { rect })
        })?;

        Ok(())
    }

    pub(crate) fn set_stencil_actions(
        self,
        triangle_face: Context3DTriangleFace,
        compare_mode: Context3DCompareMode,
        on_both_pass: Context3DStencilAction,
        on_depth_fail: Context3DStencilAction,
        on_depth_pass_stencil_fail: Context3DStencilAction,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetStencilActions {
                triangle_face,
                compare_mode,
                on_both_pass,
                on_depth_fail,
                on_depth_pass_stencil_fail,
            })
        })?;

        Ok(())
    }

    pub(crate) fn set_stencil_reference_value(
        self,
        reference_value: u32,
        read_mask: u32,
        write_mask: u32,
    ) -> Result<(), ruffle_render::error::RenderFailure> {
        self.with_context_3d_result(|ctx| {
            ctx.process_command(Context3DCommand::SetStencilReferenceValue {
                reference_value,
                read_mask,
                write_mask,
            })
        })?;

        Ok(())
    }
}

#[derive(Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct Context3DData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc>,

    #[collect(require_static)]
    render_context: Cell<Option<Box<dyn Context3D>>>,

    stage3d: Stage3DObject<'gc>,
}

impl<'gc> TObject<'gc> for Context3DObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }
}

impl std::fmt::Debug for Context3DObject<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Context3D")
    }
}

// This would ideally be placed closer to the actual usage, but
// we don't have stub support in 'render' crates
fn check_texture_stub(activation: &mut Activation<'_, '_>, format: Context3DTextureFormat) {
    match format {
        Context3DTextureFormat::BgrPacked => {
            avm2_stub_method!(
                activation,
                "flash.display3D.Context3D",
                "createTexture",
                "with BgrPacked"
            );
        }
        Context3DTextureFormat::Compressed => {
            avm2_stub_method!(
                activation,
                "flash.display3D.Context3D",
                "createTexture",
                "with Compressed"
            );
        }
        _ => {}
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    use ruffle_render::backend::{Context3DProfile, IndexBuffer, ShaderModule, VertexBuffer};
    use ruffle_render::bitmap::BitmapHandle;
    use ruffle_render::error::{
        Error as RenderError, RenderFailure, RenderFailureKind, RendererHealth,
    };
    use std::cell::RefCell;

    struct FailingTextures {
        health: RendererHealth,
        calls: Rc<Cell<usize>>,
        failure: Option<RenderFailure>,
    }
    impl FailingTextures {
        fn allocate(&self) -> Result<Rc<dyn Texture>, RenderError> {
            self.calls.set(self.calls.get() + 1);
            Err(self.failure.clone().map_or_else(
                || RenderError::Unimplemented("unsupported format".into()),
                RenderError::from,
            ))
        }
    }
    impl Context3D for FailingTextures {
        fn health(&self) -> &RendererHealth {
            &self.health
        }
        fn profile(&self) -> Context3DProfile {
            Context3DProfile::Baseline
        }
        fn bitmap_handle(&self) -> BitmapHandle {
            unreachable!()
        }
        fn back_buffer_handle(&self) -> Option<BitmapHandle> {
            None
        }
        fn back_buffer_size(&self) -> Option<(u32, u32)> {
            None
        }
        fn buffers_cleared(&self) -> bool {
            false
        }
        fn should_render(&self) -> bool {
            false
        }
        fn disposed_index_buffer_handle(&self) -> Rc<dyn IndexBuffer> {
            unreachable!()
        }
        fn disposed_vertex_buffer_handle(&self) -> Rc<dyn VertexBuffer> {
            unreachable!()
        }
        fn create_index_buffer(
            &mut self,
            _: BufferUsage,
            _: u32,
        ) -> Result<Box<dyn IndexBuffer>, RenderFailure> {
            unreachable!()
        }
        fn create_vertex_buffer(
            &mut self,
            _: BufferUsage,
            _: u32,
            _: u8,
        ) -> Result<Rc<dyn VertexBuffer>, RenderFailure> {
            unreachable!()
        }
        fn create_texture(
            &mut self,
            _: u32,
            _: u32,
            _: Context3DTextureFormat,
            _: bool,
            _: u32,
        ) -> Result<Rc<dyn Texture>, RenderError> {
            self.allocate()
        }
        fn create_cube_texture(
            &mut self,
            _: u32,
            _: Context3DTextureFormat,
            _: bool,
            _: u32,
        ) -> Result<Rc<dyn Texture>, RenderError> {
            self.allocate()
        }
        fn upload_shaders(
            &mut self,
            _: &RefCell<Option<Rc<dyn ShaderModule>>>,
            _: Vec<u8>,
            _: Vec<u8>,
        ) -> Result<(), AgalError> {
            unreachable!()
        }
        fn process_command(&mut self, _: Context3DCommand<'_>) -> Result<(), RenderFailure> {
            unreachable!()
        }
        fn present(&mut self) {
            unreachable!()
        }
    }

    #[test]
    fn texture_allocation_failures_latch_health_and_are_never_replayed() {
        for cube in [false, true] {
            let player = crate::PlayerBuilder::new().build();
            let mut player = player.lock().unwrap();
            let calls = Rc::new(Cell::new(0));
            player.mutate_with_update_context(|context| {
                let stage = Stage3DObject::new(context);
                let render_context = FailingTextures {
                    health: context.renderer.health().clone(),
                    calls: calls.clone(),
                    failure: Some(RenderFailure::new(
                        RenderFailureKind::OutOfMemory,
                        "texture allocation fault",
                    )),
                };
                let object =
                    Context3DObject::from_context(context, Box::new(render_context), stage);
                let mut activation = Activation::from_nothing(context);
                for _ in 0..2 {
                    let class = activation.avm2().classes().texture;
                    let result = if cube {
                        object.create_cube_texture(
                            8,
                            Context3DTextureFormat::Bgra,
                            false,
                            0,
                            &mut activation,
                        )
                    } else {
                        object.create_texture(
                            8,
                            8,
                            Context3DTextureFormat::Bgra,
                            false,
                            0,
                            class,
                            &mut activation,
                        )
                    };
                    assert!(result.is_err());
                }
            });
            assert_eq!(calls.get(), 1);
            assert_eq!(
                player.render_failure().unwrap().kind(),
                RenderFailureKind::OutOfMemory
            );
            assert!(player.render().is_err());
        }
    }

    #[test]
    fn unsupported_texture_capability_does_not_latch_a_resource_failure() {
        let player = crate::PlayerBuilder::new().build();
        let mut player = player.lock().unwrap();
        player.mutate_with_update_context(|context| {
            let stage = Stage3DObject::new(context);
            let render_context = FailingTextures {
                health: context.renderer.health().clone(),
                calls: Default::default(),
                failure: None,
            };
            let object = Context3DObject::from_context(context, Box::new(render_context), stage);
            let mut activation = Activation::from_nothing(context);
            assert!(
                object
                    .create_cube_texture(8, Context3DTextureFormat::Bgra, false, 0, &mut activation)
                    .is_err()
            );
        });
        assert!(player.render_failure().is_none());
    }
}
