use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::error::{Error2004Type, make_error_2004, make_error_2030};
use crate::avm2::function::FunctionArgs;
use crate::avm2::object::TObject as _;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;
use crate::pixel_bender::PixelBenderTypeExt;
use crate::string::AvmString;

use ruffle_macros::istr;
use ruffle_render::pixel_bender::{
    OUT_COORD_NAME, PixelBenderParam, PixelBenderParamQualifier, PixelBenderParsingError,
    parse_shader,
};

use super::shader_parameter::make_shader_parameter;

pub use crate::avm2::object::shader_data_allocator;

/// Implements `ShaderData._setByteCode`, which is called from the constructor
pub fn _set_byte_code<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    activation.context.renderer.health().check()?;
    let this = this.as_object().unwrap();

    let bytecode = args.get_object(activation, 0, "bytecode")?;
    let bytecode = bytecode.as_bytearray().unwrap();
    let shader = parse_shader(bytecode.bytes(), true).map_err(|err| {
        tracing::warn!("Failed to parse a Pixel Bender shader: {err}");
        match err {
            PixelBenderParsingError::IoError(_) => make_error_2030(activation),
            _ => make_error_2004(activation, Error2004Type::ArgumentError),
        }
    })?;

    for meta in &shader.metadata {
        let name = AvmString::new_utf8(activation.gc(), &meta.key);
        // Top-level metadata appears to turn `TInt` into a plain integer value,
        // rather than a single-element array.
        let value = meta.value.as_avm2_value(activation.context, true)?;
        this.set_dynamic_property(name, value, activation.gc());
    }
    this.set_dynamic_property(
        istr!("name"),
        AvmString::new_utf8(activation.gc(), &shader.name).into(),
        activation.gc(),
    );

    let mut normal_index = 0;
    let mut texture_index = 0;

    for param in &shader.params {
        let (name, index) = match &param {
            PixelBenderParam::Normal {
                name, qualifier, ..
            } => {
                // Neither of these show up in Flash Player
                if name == OUT_COORD_NAME || matches!(qualifier, PixelBenderParamQualifier::Output)
                {
                    continue;
                }
                let index = normal_index;
                normal_index += 1;
                (name, index)
            }
            PixelBenderParam::Texture { name, .. } => {
                let index = texture_index;
                texture_index += 1;
                (name, index)
            }
        };

        let name = AvmString::new_utf8(activation.gc(), name);
        let param_obj = make_shader_parameter(activation, param, index)?;
        this.set_dynamic_property(name, param_obj, activation.gc());
    }

    let shader_handle = activation
        .context
        .renderer
        .compile_pixelbender_shader(shader)
        .map_err(|error| {
            activation
                .context
                .renderer
                .health()
                .record(error.into_render_failure())
        })?;
    activation.context.renderer.health().check()?;

    this.as_shader_data()
        .unwrap()
        .set_pixel_bender_shader(shader_handle);
    Ok(Value::Undefined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_bytecode_allocation_failure_is_terminal_without_panic_or_replay() {
        use crate::avm2::{bytearray::ByteArrayStorage, object::ByteArrayObject};
        use ruffle_render::error::{RenderFailure, RenderFailureKind};
        let mut renderer = crate::bitmap::symbol_pixels::tests::renderer(false);
        renderer.allocation_failure = Some(RenderFailure::new(
            RenderFailureKind::OutOfMemory,
            "shader allocation fault",
        ));
        let calls = renderer.allocation_calls.clone();
        let player = crate::PlayerBuilder::new()
            .with_movie(crate::tag_utils::SwfMovieData::empty(10, None))
            .with_renderer(renderer)
            .build();
        let mut player = player.lock().unwrap();
        player.mutate_with_update_context(|context| {
            let bytes = ByteArrayStorage::from_vec(
                context,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tests/tests/swfs/avm2/pixelbender_logicalnot/logicalnot.pbj"
                ))
                .to_vec(),
            );
            let bytes = ByteArrayObject::from_storage(context, bytes);
            let mut activation = Activation::from_nothing(context);
            let domain = activation.avm2().playerglobals_domain();
            let name = domain
                .get_defined_names()
                .into_iter()
                .find(|name| name.local_name().as_wstr() == b"ShaderData")
                .expect("ShaderData exists in playerglobals");
            let class = domain
                .get_defined_value(&mut activation, name)
                .unwrap_or_else(|error| panic!("{}", error.to_string(&mut activation)))
                .as_object()
                .unwrap()
                .as_class_object()
                .unwrap();
            let shader = shader_data_allocator(class, &mut activation).unwrap();
            for _ in 0..2 {
                let args = [bytes.into()];
                assert!(
                    _set_byte_code(
                        &mut activation,
                        shader.into(),
                        FunctionArgs::from_slice(&args)
                    )
                    .is_err()
                );
            }
            assert!(
                shader
                    .as_shader_data()
                    .unwrap()
                    .pixel_bender_shader()
                    .is_none()
            );
        });
        assert_eq!(calls.get(), 1);
        assert_eq!(
            player.render_failure().unwrap().kind(),
            RenderFailureKind::OutOfMemory
        );
        assert!(player.render().is_err());
    }
}
