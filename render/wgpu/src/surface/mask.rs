use enum_map::Enum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum MaskMode {
    NoMask,
    DrawMaskStencil,
    DrawMaskedContent,
    ClearMaskStencil,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskState {
    pub mode: MaskMode,
    pub count: u32,
}

impl MaskState {
    pub fn empty() -> Self {
        MaskState {
            mode: MaskMode::NoMask,
            count: 0,
        }
    }

    pub fn apply_stencil_reference(self, render_pass: &mut crate::gpu_resources::RenderPass) {
        match self.mode {
            MaskMode::NoMask => {}
            MaskMode::DrawMaskStencil => {
                render_pass.set_stencil_reference(self.count - 1);
            }
            MaskMode::DrawMaskedContent => {
                render_pass.set_stencil_reference(self.count);
            }
            MaskMode::ClearMaskStencil => {
                render_pass.set_stencil_reference(self.count);
            }
        }
    }

    pub fn push_mask(&mut self) {
        debug_assert!(self.mode == MaskMode::NoMask || self.mode == MaskMode::DrawMaskedContent);
        self.count += 1;
        self.mode = MaskMode::DrawMaskStencil;
    }

    pub fn activate_mask(&mut self) {
        debug_assert!(self.count > 0 && self.mode == MaskMode::DrawMaskStencil);
        self.mode = MaskMode::DrawMaskedContent;
    }

    pub fn deactivate_mask(&mut self) {
        debug_assert!(self.count > 0 && self.mode == MaskMode::DrawMaskedContent);
        self.mode = MaskMode::ClearMaskStencil;
    }

    pub fn pop_mask(&mut self) {
        debug_assert!(self.count > 0 && self.mode == MaskMode::ClearMaskStencil);
        self.count -= 1;
        if self.count == 0 {
            self.mode = MaskMode::NoMask;
        } else {
            self.mode = MaskMode::DrawMaskedContent;
        };
    }
}

impl MaskMode {
    pub fn pipeline_stencil(self) -> wgpu::DepthStencilState {
        let face = wgpu::StencilFaceState {
            compare: if self == Self::NoMask {
                wgpu::CompareFunction::Always
            } else {
                wgpu::CompareFunction::Equal
            },
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: match self {
                Self::DrawMaskStencil => wgpu::StencilOperation::IncrementClamp,
                Self::ClearMaskStencil => wgpu::StencilOperation::DecrementClamp,
                _ => wgpu::StencilOperation::Keep,
            },
        };
        wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Stencil8,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState {
                front: face,
                back: face,
                read_mask: !0,
                write_mask: !0,
            },
            bias: Default::default(),
        }
    }
    pub fn color_writes(self) -> wgpu::ColorWrites {
        match self {
            Self::DrawMaskStencil | Self::ClearMaskStencil => wgpu::ColorWrites::empty(),
            _ => wgpu::ColorWrites::ALL,
        }
    }
}
