//! Upload bytes and ordering cross the command boundary. The context retains
//! destinations only until their uploads have been recorded in its encoder.
use crate::gpu_resources::{Buffer, CommandEncoder, StagingBelt, Texture};
use crate::gpu_runtime::{RenderFailure, RenderFailureKind, ResourceId};
use fnv::FnvHashMap;
use std::collections::VecDeque;
use std::num::NonZeroU64;

#[derive(Clone)]
pub(super) enum UploadTarget {
    Buffer(Buffer),
    Texture(Texture),
}

impl UploadTarget {
    fn id(&self) -> ResourceId {
        match self {
            Self::Buffer(buffer) => buffer.lease().id(),
            Self::Texture(texture) => texture.lease().id(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum UploadLayout {
    Buffer {
        offset: u64,
    },
    Texture {
        layer: u32,
        size: wgpu::Extent3d,
        bgra: bool,
    },
}

impl UploadLayout {
    pub(super) fn encode(
        self,
        target: &UploadTarget,
        data: &[u8],
        belt: &mut StagingBelt,
        encoder: &mut CommandEncoder,
    ) -> Result<(), RenderFailure> {
        match (self, target) {
            (Self::Buffer { offset }, UploadTarget::Buffer(buffer)) => {
                if let Some(length) = NonZeroU64::new(data.len() as u64) {
                    belt.write_buffer(encoder, buffer, offset, length)?
                        .slice(..data.len())
                        .copy_from_slice(data);
                }
            }
            (Self::Texture { layer, size, bgra }, UploadTarget::Texture(texture)) => {
                belt.write_texture(
                    encoder,
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    bgra,
                    size,
                )?;
            }
            _ => {
                return Err(RenderFailure::new(
                    RenderFailureKind::Validation,
                    "Stage3D upload target type changed",
                ));
            }
        }
        Ok(())
    }
}

struct UploadPacket {
    sequence: u64,
    destination: ResourceId,
    layout: UploadLayout,
    data: Box<[u8]>,
}

struct Binding {
    target: UploadTarget,
    remaining: usize,
}

#[derive(Default)]
pub(super) struct UploadQueue {
    issued: u64,
    applied: u64,
    pending: VecDeque<UploadPacket>,
    bindings: FnvHashMap<ResourceId, Binding>,
    failure: Option<RenderFailure>,
}

impl UploadQueue {
    pub(super) fn enqueue(
        &mut self,
        target: UploadTarget,
        layout: UploadLayout,
        data: &[u8],
    ) -> Result<(), RenderFailure> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let sequence = self.issued.checked_add(1).ok_or_else(|| {
            RenderFailure::new(
                RenderFailureKind::Internal,
                "Stage3D upload sequence exhausted",
            )
        })?;
        let destination = target.id();
        self.bindings
            .entry(destination)
            .or_insert(Binding {
                target,
                remaining: 0,
            })
            .remaining += 1;
        self.pending.push_back(UploadPacket {
            sequence,
            destination,
            layout,
            data: data.into(),
        });
        self.issued = sequence;
        Ok(())
    }

    pub(super) fn flush(
        &mut self,
        belt: &mut StagingBelt,
        encoder: &mut CommandEncoder,
    ) -> Result<(), RenderFailure> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        while let Some(packet) = self.pending.pop_front() {
            let binding = self
                .bindings
                .get_mut(&packet.destination)
                .expect("queued upload pins its destination");
            if let Err(error) = packet
                .layout
                .encode(&binding.target, &packet.data, belt, encoder)
            {
                self.failure = Some(error.clone());
                self.pending.clear();
                self.bindings.clear();
                return Err(error);
            }
            self.applied = packet.sequence;
            binding.remaining -= 1;
            if binding.remaining == 0 {
                self.bindings.remove(&packet.destination);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn applied(&self) -> u64 {
        self.applied
    }
}

const _: fn() = || {
    fn transport<T: Send + 'static>() {}
    transport::<UploadPacket>();
};
