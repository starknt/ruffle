//! Upload bytes and ordering cross the command boundary. The context retains
//! destinations only until their uploads have been recorded in its encoder.
use crate::gpu_resources::{Buffer, CommandEncoder, StagingBelt, Texture};
use crate::gpu_runtime::{GpuRuntime, OwnedPayload, RenderFailure, RenderFailureKind, ResourceId};
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
                let bytes = data.len() as u64;
                if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
                    || !bytes.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
                    || offset
                        .checked_add(bytes)
                        .is_none_or(|end| end > buffer.size())
                {
                    return Err(RenderFailure::new(
                        RenderFailureKind::Validation,
                        "invalid Stage3D buffer upload range",
                    ));
                }
                let page = belt.page_size();
                let mut position = offset;
                for chunk in data.chunks(page as usize) {
                    encoder.flush_uploads_if_needed(page)?;
                    let length = NonZeroU64::new(chunk.len() as u64).expect("nonempty chunk");
                    belt.write_buffer(encoder, buffer, position, length)?
                        .copy_from_slice(chunk);
                    position += length.get();
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
    payload: OwnedPayload,
}

struct Binding {
    target: UploadTarget,
    remaining: usize,
}

pub(super) struct UploadQueue {
    runtime: GpuRuntime,
    issued: u64,
    applied: u64,
    pending: VecDeque<UploadPacket>,
    bindings: FnvHashMap<ResourceId, Binding>,
    failure: Option<RenderFailure>,
}

impl UploadQueue {
    pub(super) fn new(runtime: GpuRuntime) -> Self {
        Self {
            runtime,
            issued: 0,
            applied: 0,
            pending: Default::default(),
            bindings: Default::default(),
            failure: None,
        }
    }

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
        if destination.device_generation() != self.runtime.generation() {
            return Err(RenderFailure::new(
                RenderFailureKind::Validation,
                "Stage3D upload targets another device",
            ));
        }
        let payload = self
            .runtime
            .reserve_payload(crate::gpu_runtime::PayloadKind::Upload, data.len() as u64)?;
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
            payload,
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
        while let Some(mut packet) = self.pending.pop_front() {
            packet.payload.begin_encoding();
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
