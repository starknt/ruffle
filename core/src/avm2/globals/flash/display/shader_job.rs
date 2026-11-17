use crate::avm2::Avm2;
use crate::avm2::bytearray::Endian;
use crate::avm2::error::{Error2004Type, make_error_2004, make_error_2162, make_error_2165};
use crate::avm2::function::FunctionArgs;
use crate::avm2::globals::slots::{
    flash_display_shader as shader_slots, flash_display_shader_input as shader_input_slots,
    flash_display_shader_job as shader_job_slots,
    flash_display_shader_parameter as shader_parameter_slots,
};
use crate::avm2::object::EventObject;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::{Activation, Error, Object, TObject as _, Value};
use crate::pixel_bender::PixelBenderTypeExt;
use crate::string::AvmString;

use gc_arena::Collect;

use std::collections::VecDeque;
use std::time::Instant;

/// The lifecycle of a non-blocking ShaderJob. The queue is deliberately kept
/// independent of AVM values so it can be tested without a renderer or GC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncShaderJobPhase {
    Queued,
    Submitted,
    WaitingReadback,
    Ready,
    Canceled,
}

#[derive(Debug, Default, Collect)]
#[collect(no_drop)]
pub struct ShaderJobQueue {
    next_id: u64,
    pending: VecDeque<u64>,
}

#[derive(Collect)]
#[collect(no_drop)]
pub struct PendingShaderJob<'gc> {
    pub id: u64,
    pub job: Object<'gc>,
    pub target: Object<'gc>,
    #[collect(require_static)]
    pub shader_handle: PixelBenderShaderHandle,
    #[collect(require_static)]
    pub arguments: Vec<PixelBenderShaderArgument<'static>>,
    #[collect(require_static)]
    pub pixel_bender_target: PixelBenderTarget,
    #[collect(require_static)]
    pub phase: AsyncShaderJobPhase,
    #[collect(require_static)]
    pub pending: Option<PixelBenderPending>,
}

impl ShaderJobQueue {
    pub fn enqueue(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.pending.push_back(id);
        id
    }

    pub fn pop_for_submission(&mut self) -> Option<u64> {
        self.pending.pop_front()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn cancel(&mut self, id: u64) -> bool {
        let old_len = self.pending.len();
        self.pending.retain(|queued| *queued != id);
        old_len != self.pending.len()
    }
}

use ruffle_render::backend::{PixelBenderOutput, PixelBenderPending, PixelBenderTarget};
use ruffle_render::bitmap::PixelRegion;
use ruffle_render::pixel_bender::{
    OUT_COORD_NAME, PixelBenderMetadata, PixelBenderParam, PixelBenderParamQualifier,
    PixelBenderShaderHandle, PixelBenderType, PixelBenderTypeOpcode,
};
use ruffle_render::pixel_bender_support::{
    FloatPixelData, ImageInputTexture, PixelBenderShaderArgument,
};

/// Move jobs queued during the previous frame into the submission list.
/// This is called before ActionScript executes, ensuring a new start(false)
/// cannot complete in the same frame in which it was requested.
pub fn begin_frame(context: &mut crate::context::UpdateContext<'_>) {
    let mut pending = std::mem::take(context.pending_shader_jobs);
    let pending_depth = pending.len();
    for job in &mut pending {
        let submitted_at = Instant::now();
        job.phase = AsyncShaderJobPhase::Submitted;
        ruffle_render::profile::async_event(
            "shaderJobSubmitted",
            job.id,
            pending_depth,
            submitted_at.elapsed().as_nanos() as u64,
            pixel_bender_output_bytes(&job.pixel_bender_target),
            None,
        );
    }
    context.active_shader_jobs.append(&mut pending);
}

/// Run queued jobs at the end of a frame and dispatch their completion events.
/// All AVM objects and renderer access stay on the player thread.
pub fn process_post_frame(context: &mut crate::context::UpdateContext<'_>) {
    if context.renderer.health().check().is_err() {
        context.pending_shader_jobs.clear();
        context.active_shader_jobs.clear();
        return;
    }
    let mut jobs = std::mem::take(context.active_shader_jobs);
    let jobs_this_frame = jobs.len();
    for _ in 0..jobs_this_frame {
        if context.renderer.health().check().is_err() {
            context.pending_shader_jobs.clear();
            context.active_shader_jobs.clear();
            return;
        }
        let Some(mut job) = jobs.pop_front() else {
            break;
        };
        if let Some(mut pending) = job.pending.take() {
            let polled_at = Instant::now();
            match context.renderer.poll_pixelbender_shader(&mut pending) {
                Ok(Some(output)) => {
                    if context.renderer.health().check().is_err() {
                        context.pending_shader_jobs.clear();
                        context.active_shader_jobs.clear();
                        return;
                    }
                    job.phase = AsyncShaderJobPhase::Ready;
                    ruffle_render::profile::async_event(
                        "shaderJobCompleted",
                        job.id,
                        context.active_shader_jobs.len(),
                        polled_at.elapsed().as_nanos() as u64,
                        pixel_bender_output_bytes(&job.pixel_bender_target),
                        None,
                    );
                    if let Err(error) = finish_shader_job(context, job.job, job.target, output) {
                        tracing::error!(
                            job_id = job.id,
                            ?error,
                            "Asynchronous ShaderJob result failed"
                        );
                    }
                }
                Ok(None) => {
                    job.phase = AsyncShaderJobPhase::WaitingReadback;
                    job.pending = Some(pending);
                    context.active_shader_jobs.push_back(job);
                }
                Err(error) => {
                    tracing::error!(
                        job_id = job.id,
                        ?error,
                        "Asynchronous ShaderJob readback failed"
                    );
                    ruffle_render::profile::async_event(
                        "shaderJobCanceled",
                        job.id,
                        context.active_shader_jobs.len(),
                        0,
                        pixel_bender_output_bytes(&job.pixel_bender_target),
                        Some("readback"),
                    );
                    job.phase = AsyncShaderJobPhase::Canceled;
                }
            }
            continue;
        }

        let submitted_at = Instant::now();
        match context.renderer.submit_pixelbender_shader(
            job.shader_handle.clone(),
            &job.arguments,
            &job.pixel_bender_target,
        ) {
            Ok(pending) => {
                job.phase = AsyncShaderJobPhase::WaitingReadback;
                job.pending = Some(pending);
                context.active_shader_jobs.push_back(job);
            }
            Err(error) => {
                tracing::error!(
                    job_id = job.id,
                    ?error,
                    "Asynchronous ShaderJob submission failed"
                );
                ruffle_render::profile::async_event(
                    "shaderJobCanceled",
                    job.id,
                    context.active_shader_jobs.len(),
                    submitted_at.elapsed().as_nanos() as u64,
                    pixel_bender_output_bytes(&job.pixel_bender_target),
                    Some("submit"),
                );
                job.phase = AsyncShaderJobPhase::Canceled;
            }
        }
    }
}

fn pixel_bender_output_bytes(target: &PixelBenderTarget) -> usize {
    match target {
        PixelBenderTarget::Bitmap(_) => 0,
        PixelBenderTarget::Bytes { width, height } => (*width as usize)
            .saturating_mul(*height as usize)
            .saturating_mul(16),
    }
}

fn finish_shader_job<'gc>(
    context: &mut crate::context::UpdateContext<'gc>,
    job: Object<'gc>,
    target: Object<'gc>,
    output: PixelBenderOutput,
) -> Result<(), Error<'gc>> {
    match output {
        PixelBenderOutput::Bitmap(sync_handle) => {
            let bitmap = target
                .as_bitmap_data()
                .expect("Bitmap ShaderJob target disappeared");
            bitmap.mark_gpu_dirty(
                context.gc_context,
                sync_handle,
                PixelRegion::for_whole_size(bitmap.width(), bitmap.height()),
            );
        }
        PixelBenderOutput::Bytes(pixels) => {
            if let Some(mut bytearray) = target.as_bytearray_mut() {
                bytearray.write_at(&pixels, 0).unwrap();
            } else if let Some(mut vector) = target.as_vector_storage_mut(context.gc_context) {
                let new_values = bytemuck::cast_slice::<u8, f32>(&pixels)
                    .iter()
                    .map(|p| Value::from(*p as f64));
                vector.replace_storage_with_iter(new_values);
            } else {
                let mut activation = Activation::from_nothing(context);
                return Err(make_error_2004(
                    &mut activation,
                    Error2004Type::ArgumentError,
                ));
            }
        }
    }

    let mut activation = Activation::from_nothing(context);

    let (bitmap_data, byte_array, vector) = if target.as_bitmap_data().is_some() {
        (Some(target), None, None)
    } else if target.as_bytearray().is_some() {
        (None, Some(target), None)
    } else if target.as_vector_storage().is_some() {
        (None, None, Some(target))
    } else {
        (None, None, None)
    };
    let complete_evt =
        EventObject::shader_event(&mut activation, "complete", bitmap_data, byte_array, vector);
    Avm2::dispatch_event(context, complete_evt, job);
    Ok(())
}

/// Get the default value for a shader parameter from its metadata.
/// If no default is found, returns a empty value of the appropriate type.
fn get_default_shader_param_value(
    metadata: &[PixelBenderMetadata],
    param_type: PixelBenderTypeOpcode,
) -> PixelBenderType {
    for meta in metadata {
        if meta.key == "defaultValue" {
            return meta.value.clone();
        }
    }

    match param_type {
        PixelBenderTypeOpcode::TFloat => PixelBenderType::TFloat(0.0),
        PixelBenderTypeOpcode::TFloat2 => PixelBenderType::TFloat2(0.0, 0.0),
        PixelBenderTypeOpcode::TFloat3 => PixelBenderType::TFloat3(0.0, 0.0, 0.0),
        PixelBenderTypeOpcode::TFloat4 => PixelBenderType::TFloat4(0.0, 0.0, 0.0, 0.0),
        PixelBenderTypeOpcode::TFloat2x2 => PixelBenderType::TFloat2x2([0.0; 4]),
        PixelBenderTypeOpcode::TFloat3x3 => PixelBenderType::TFloat3x3([0.0; 9]),
        PixelBenderTypeOpcode::TFloat4x4 => PixelBenderType::TFloat4x4([0.0; 16]),
        PixelBenderTypeOpcode::TInt => PixelBenderType::TInt(0),
        PixelBenderTypeOpcode::TInt2 => PixelBenderType::TInt2(0, 0),
        PixelBenderTypeOpcode::TInt3 => PixelBenderType::TInt3(0, 0, 0),
        PixelBenderTypeOpcode::TInt4 => PixelBenderType::TInt4(0, 0, 0, 0),
        PixelBenderTypeOpcode::TString => PixelBenderType::TString(String::new()),
        PixelBenderTypeOpcode::TBool => PixelBenderType::TBool(0),
        PixelBenderTypeOpcode::TBool2 => PixelBenderType::TBool2(0, 0),
        PixelBenderTypeOpcode::TBool3 => PixelBenderType::TBool3(0, 0, 0),
        PixelBenderTypeOpcode::TBool4 => PixelBenderType::TBool4(0, 0, 0, 0),
    }
}

pub fn get_shader_args<'gc>(
    shader_obj: Object<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<
    (
        PixelBenderShaderHandle,
        Vec<PixelBenderShaderArgument<'static>>,
    ),
    Error<'gc>,
> {
    // FIXME - determine what errors Flash Player throws here
    // instead of using `expect`
    let shader_data = shader_obj
        .get_slot(shader_slots::_DATA)
        .as_object()
        .expect("Missing ShaderData object")
        .as_shader_data()
        .expect("ShaderData object is not a ShaderData instance");

    let shader_handle = shader_data.pixel_bender_shader();
    let shader_handle = shader_handle
        .as_ref()
        .expect("ShaderData object has no shader");
    let shader = shader_handle.0.parsed_shader();

    let args = shader
        .params
        .iter()
        .enumerate()
        .filter(|(_, param)| {
            !matches!(
                param,
                PixelBenderParam::Normal {
                    qualifier: PixelBenderParamQualifier::Output,
                    ..
                }
            )
        })
        .map(|(index, param)| {
            match param {
                PixelBenderParam::Normal {
                    param_type,
                    name,
                    metadata,
                    ..
                } => {
                    if name == OUT_COORD_NAME {
                        // Pass in a dummy value - this will be ignored in favor of the actual pixel coordinate
                        return Ok(PixelBenderShaderArgument::ValueInput {
                            index: index as u8,
                            value: PixelBenderType::TFloat2(f32::NAN, f32::NAN),
                        });
                    }
                    let shader_param = shader_data
                        .get_dynamic_property(AvmString::new_utf8(activation.gc(), name))
                        .expect("Missing normal property");

                    let pb_val = if let Some(shader_param) = shader_param.as_object()
                        && shader_param.is_of_type(
                            activation
                                .avm2()
                                .classes()
                                .shaderparameter
                                .inner_class_definition(),
                        ) {
                        let value = shader_param.get_slot(shader_parameter_slots::_VALUE);
                        PixelBenderType::from_avm2_value(activation, value, param_type)?
                    } else {
                        // The ShaderParameter was replaced with a primitive or non-ShaderParameter object.
                        // Flash ignores this and uses the default value from shader metadata.
                        get_default_shader_param_value(metadata, *param_type)
                    };

                    Ok(PixelBenderShaderArgument::ValueInput {
                        index: index as u8,
                        value: pb_val,
                    })
                }
                PixelBenderParam::Texture {
                    index,
                    channels,
                    name,
                } => {
                    let shader_input = shader_data
                        .get_dynamic_property(AvmString::new_utf8(activation.gc(), name))
                        .expect("Missing property")
                        .as_object()
                        .expect("Shader input is not an object");

                    if !shader_input.is_of_type(
                        activation
                            .avm2()
                            .classes()
                            .shaderinput
                            .inner_class_definition(),
                    ) {
                        panic!("Expected shader input to be of class ShaderInput");
                    }

                    let input = shader_input.get_slot(shader_input_slots::_INPUT);

                    let width = shader_input.get_slot(shader_input_slots::_WIDTH).as_u32();
                    let height = shader_input.get_slot(shader_input_slots::_HEIGHT).as_u32();

                    let input_channels = shader_input
                        .get_slot(shader_input_slots::_CHANNELS)
                        .as_u32();

                    assert_eq!(*channels as u32, input_channels);

                    let texture = if let Some(input) = input.as_object() {
                        let input_texture = if let Some(bitmap) = input.as_bitmap_data() {
                            ImageInputTexture::Bitmap(
                                bitmap
                                    .bitmap_handle(activation.gc(), activation.context.renderer)?,
                            )
                        } else if let Some(byte_array) = input.as_bytearray() {
                            assert_eq!(byte_array.endian(), Endian::Little);

                            let (bytes, _) = byte_array.bytes().as_chunks::<4>();
                            let floats = bytemuck::cast_slice::<[u8; 4], f32>(bytes);

                            make_float_texture(
                                activation,
                                name,
                                floats,
                                width,
                                height,
                                input_channels,
                            )?
                        } else if let Some(vector) = input.as_vector_storage() {
                            let values: &[Value<'gc>] = vector.storage().as_ref();

                            make_float_texture(
                                activation,
                                name,
                                values,
                                width,
                                height,
                                input_channels,
                            )?
                        } else {
                            panic!("Unexpected input object {input:?}");
                        };
                        Some(input_texture)
                    } else {
                        // Null input
                        None
                    };

                    Ok(PixelBenderShaderArgument::ImageInput {
                        index: *index,
                        channels: *channels,
                        name: name.clone(),
                        texture,
                    })
                }
            }
        })
        .collect::<Result<Vec<PixelBenderShaderArgument<'_>>, Error<'gc>>>()?;
    Ok((shader_handle.clone(), args))
}

trait PixelSource {
    fn collect<const N: usize>(&self, num_pixels: usize) -> Option<Vec<[f32; N]>>;
}

impl PixelSource for &[f32] {
    fn collect<const N: usize>(&self, num_pixels: usize) -> Option<Vec<[f32; N]>> {
        let (floats, _) = self.as_chunks::<N>();
        Some(floats.get(..num_pixels)?.to_vec())
    }
}

impl<'gc> PixelSource for &[Value<'gc>] {
    fn collect<const N: usize>(&self, num_pixels: usize) -> Option<Vec<[f32; N]>> {
        let (chunks, _) = self.as_chunks::<N>();
        Some(
            chunks
                .get(..num_pixels)?
                .iter()
                .map(|vals| vals.map(|val| val.as_f64() as f32))
                .collect(),
        )
    }
}

fn make_float_texture<'gc, S: PixelSource>(
    activation: &mut Activation<'_, 'gc>,
    shader_name: &str,
    source: S,
    width: u32,
    height: u32,
    input_channels: u32,
) -> Result<ImageInputTexture<'static>, Error<'gc>> {
    let num_pixels = (width * height) as usize;
    let err = || make_error_2165(activation, shader_name);

    let data = match input_channels {
        1 => FloatPixelData::R(source.collect::<1>(num_pixels).ok_or_else(err)?),
        2 => FloatPixelData::Rg(source.collect::<2>(num_pixels).ok_or_else(err)?),
        3 => FloatPixelData::Rgb(source.collect::<3>(num_pixels).ok_or_else(err)?),
        4 => FloatPixelData::Rgba(source.collect::<4>(num_pixels).ok_or_else(err)?),
        _ => panic!("Unexpected number of channels: {input_channels}"),
    };

    Ok(ImageInputTexture::Floats {
        width,
        height,
        data,
    })
}

/// Implements `ShaderJob.start`.
pub fn start<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();

    let wait_for_completion = args.get_bool(0);
    let shader = this
        .get_slot(shader_job_slots::_SHADER)
        .as_object()
        .expect("Missing Shader object");

    let (shader_handle, arguments) = get_shader_args(shader, activation)?;

    let target = this
        .get_slot(shader_job_slots::_TARGET)
        .as_object()
        .expect("ShaderJob.target is not an object");

    let output_width = this.get_slot(shader_job_slots::_WIDTH).as_u32();

    let output_height = this.get_slot(shader_job_slots::_HEIGHT).as_u32();

    let pixel_bender_target = if let Some(bitmap) = target.as_bitmap_data() {
        PixelBenderTarget::Bitmap(
            bitmap.bitmap_handle(activation.gc(), activation.context.renderer)?,
        )
    } else {
        PixelBenderTarget::Bytes {
            width: output_width,
            height: output_height,
        }
    };

    match shader_handle.0.parsed_shader().output_channels() {
        Some(3 | 4) => {}
        channels => {
            tracing::warn!(
                "Unsupported number of shader output channels: {channels:?}, expected 3 or 4"
            );
            return Err(make_error_2162(activation));
        }
    }

    if !wait_for_completion {
        let output_bytes = pixel_bender_output_bytes(&pixel_bender_target);
        let id = activation.context.shader_job_queue.enqueue();
        activation
            .context
            .pending_shader_jobs
            .push_back(PendingShaderJob {
                id,
                job: this,
                target,
                shader_handle,
                arguments,
                pixel_bender_target,
                phase: AsyncShaderJobPhase::Queued,
                pending: None,
            });
        ruffle_render::profile::async_event(
            "shaderJobQueued",
            id,
            activation.context.pending_shader_jobs.len(),
            0,
            output_bytes,
            None,
        );
        tracing::debug!(job_id = id, "ShaderJob queued for next frame");
        return Ok(Value::Undefined);
    }

    let output = activation
        .context
        .renderer
        .run_pixelbender_shader(shader_handle, &arguments, &pixel_bender_target)
        .expect("Failed to run shader");

    match output {
        PixelBenderOutput::Bitmap(sync_handle) => {
            let bitmap = target.as_bitmap_data().unwrap();
            bitmap.mark_gpu_dirty(
                activation.gc(),
                sync_handle,
                PixelRegion::for_whole_size(bitmap.width(), bitmap.height()),
            );
        }
        PixelBenderOutput::Bytes(pixels) => {
            if let Some(mut bytearray) = target.as_bytearray_mut() {
                bytearray.write_at(&pixels, 0).unwrap();
            } else if let Some(mut vector) = target.as_vector_storage_mut(activation.gc()) {
                let new_values = bytemuck::cast_slice::<u8, f32>(&pixels)
                    .iter()
                    .map(|p| Value::from(*p as f64));
                vector.replace_storage_with_iter(new_values);
            } else {
                return Err(make_error_2004(activation, Error2004Type::ArgumentError));
            }
        }
    }

    // We always execute the shader synchronously above, so the `complete`
    // event that ActionScript relies on to read the result would otherwise
    // never fire.
    let (bitmap_data, byte_array, vector) = if target.as_bitmap_data().is_some() {
        (Some(target), None, None)
    } else if target.as_bytearray().is_some() {
        (None, Some(target), None)
    } else if target.as_vector_storage().is_some() {
        (None, None, Some(target))
    } else {
        (None, None, None)
    };
    let complete_evt =
        EventObject::shader_event(activation, "complete", bitmap_data, byte_array, vector);
    Avm2::dispatch_event(activation.context, complete_evt, this);

    Ok(Value::Undefined)
}

#[cfg(test)]
mod tests {
    use super::ShaderJobQueue;

    #[test]
    fn async_jobs_are_fifo_and_cancellable_before_submission() {
        let mut queue = ShaderJobQueue::default();
        let first = queue.enqueue();
        let second = queue.enqueue();
        let third = queue.enqueue();

        assert_eq!(queue.pending_len(), 3);
        assert!(queue.cancel(second));
        assert_eq!(queue.pop_for_submission(), Some(first));
        assert_eq!(queue.pop_for_submission(), Some(third));
        assert_eq!(queue.pop_for_submission(), None);
        assert_eq!(queue.pending_len(), 0);
    }
}
