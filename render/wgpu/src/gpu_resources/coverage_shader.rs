use std::borrow::Cow;

use naga::{
    Arena, Binding, Block, Expression, Handle, Scalar, Span, Statement, StructMember, Type,
    TypeInner,
};

/// Builds the fragment variant used when additive color needs a separate coverage target.
pub(super) fn shader(
    device: &super::Device,
    source: &wgpu::ShaderModuleDescriptor<'_>,
) -> Option<wgpu::ShaderModule> {
    let module = coverage_module(&source.source)?;
    device
        .scoped(|d| {
            Ok(d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: source.label,
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            }))
        })
        .ok()
}

fn coverage_module(source: &wgpu::ShaderSource<'_>) -> Option<naga::Module> {
    let mut module = match source {
        wgpu::ShaderSource::Wgsl(text) => naga::front::wgsl::parse_str(text).ok()?,
        wgpu::ShaderSource::Naga(module) => module.as_ref().clone(),
        _ => return None,
    };
    let mut fragments = module
        .entry_points
        .iter_mut()
        .filter(|entry| entry.stage == naga::ShaderStage::Fragment)
        .peekable();
    fragments.peek()?;
    let float = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        },
        Span::UNDEFINED,
    );
    for entry in fragments {
        let function = &mut entry.function;
        let result = function.result.as_mut()?;
        if !matches!(
            module.types.get_handle(result.ty).ok()?.inner,
            TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar: Scalar::F32
            }
        ) {
            return None;
        }
        let color_binding = result.binding.take()?;
        let mut coverage_binding = color_binding.clone();
        match &mut coverage_binding {
            Binding::Location {
                location,
                blend_src: None,
                per_primitive: false,
                ..
            } if *location == 0 => *location = 1,
            _ => return None,
        }
        result.ty = module.types.insert(
            Type {
                name: Some("CoveredColor".into()),
                inner: TypeInner::Struct {
                    members: vec![
                        StructMember {
                            name: Some("color".into()),
                            ty: result.ty,
                            binding: Some(color_binding),
                            offset: 0,
                        },
                        StructMember {
                            name: Some("coverage".into()),
                            ty: float,
                            binding: Some(coverage_binding),
                            offset: 16,
                        },
                    ],
                    // The vec4 gives this struct 16-byte alignment.
                    span: 32,
                },
            },
            Span::UNDEFINED,
        );
        let one = function.expressions.append(
            Expression::Literal(naga::Literal::F32(1.0)),
            Span::UNDEFINED,
        );
        cover_returns(
            &mut function.body,
            &mut function.expressions,
            result.ty,
            one,
        )?;
    }
    Some(module)
}

fn cover_returns(
    block: &mut Block,
    expressions: &mut Arena<Expression>,
    output: Handle<Type>,
    one: Handle<Expression>,
) -> Option<()> {
    for (mut statement, span) in std::mem::take(block).span_into_iter() {
        let mut recurse = |nested: &mut Block| cover_returns(nested, expressions, output, one);
        match &mut statement {
            Statement::Block(nested) => recurse(nested)?,
            Statement::If { accept, reject, .. } => {
                recurse(accept)?;
                recurse(reject)?;
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    recurse(&mut case.body)?;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                recurse(body)?;
                recurse(continuing)?;
            }
            Statement::Return { value } => {
                let color = value.as_mut()?;
                *color = expressions.append(
                    Expression::Compose {
                        ty: output,
                        components: vec![*color, one],
                    },
                    span,
                );
                block.push(
                    Statement::Emit(naga::Range::new_from_bounds(*color, *color)),
                    span,
                );
            }
            _ => {}
        }
        block.push(statement, span);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn early_and_final_returns_write_coverage_and_discard_survives() {
        let wgsl = "@fragment fn shade(@location(0) a: vec4f, @location(1) b: vec4f) -> @location(0) vec4f {
            if a.x < 0.0 { discard; }
            if a.y < 0.0 { return b; }
            return a;
        }";
        let original = naga::front::wgsl::parse_str(wgsl).unwrap();
        for source in [
            wgpu::ShaderSource::Wgsl(wgsl.into()),
            wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(original)),
        ] {
            let module = coverage_module(&source).unwrap();
            let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap();
            let entry = &module.entry_points[0];
            assert_eq!(entry.name, "shade");
            assert!(info.get_entry_point(0).may_kill);
            let output = entry.function.result.as_ref().unwrap().ty;
            let naga::TypeInner::Struct { members, .. } = &module.types[output].inner else {
                panic!("expected color and coverage outputs")
            };
            assert_eq!(members.len(), 2);
            for (location, member) in members.iter().enumerate() {
                assert!(
                    matches!(member.binding, Some(naga::Binding::Location { location: n, .. }) if n == location as u32)
                );
            }
            let exprs = &entry.function.expressions;
            let mut colors = Vec::new();
            for (_, expr) in exprs.iter() {
                if let naga::Expression::Compose { ty, components } = expr
                    && *ty == output
                {
                    let naga::Expression::FunctionArgument(arg) = exprs[components[0]] else {
                        panic!("expected original color")
                    };
                    assert!(matches!(
                        exprs[components[1]],
                        naga::Expression::Literal(naga::Literal::F32(1.0))
                    ));
                    colors.push(arg);
                }
            }
            colors.sort_unstable();
            assert_eq!(colors, [0, 1]);
        }
    }

    #[test]
    fn unsupported_fragment_makes_the_whole_variant_unavailable() {
        let wgsl = "@fragment fn color() -> @location(0) vec4f { return vec4f(); }
                    @fragment fn depth() -> @builtin(frag_depth) f32 { return 0.5; }";
        assert!(coverage_module(&wgpu::ShaderSource::Wgsl(wgsl.into())).is_none());
    }
}
