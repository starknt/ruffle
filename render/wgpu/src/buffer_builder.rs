use bytemuck::{AnyBitPattern, NoUninit};
use std::ops::Range;

pub struct BufferBuilder {
    inner: Vec<u8>,
    align_mask: usize,
    limit: u64,
    scratch: Option<ScratchPool>,
}

#[derive(Clone, Default)]
pub struct ScratchPool(std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>>);
impl Drop for BufferBuilder {
    fn drop(&mut self) {
        if let Some(pool) = &self.scratch {
            let mut buffers = pool
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let capacity = self.inner.capacity();
            if capacity != 0
                && buffers.len() < 8
                && capacity.saturating_add(buffers.iter().map(Vec::capacity).sum::<usize>())
                    <= 2 * 1024 * 1024
            {
                let mut bytes = std::mem::take(&mut self.inner);
                bytes.clear();
                buffers.push(bytes);
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct BufferFull;

impl BufferBuilder {
    pub fn new(limits: &wgpu::Limits, alignment: u32) -> Self {
        Self {
            inner: Vec::new(),
            scratch: None,
            align_mask: if alignment > 0 {
                assert!(alignment.is_power_of_two());
                (alignment - 1) as usize
            } else {
                0
            },
            limit: limits.max_buffer_size,
        }
    }

    pub fn with_scratch(limits: &wgpu::Limits, alignment: u32, scratch: ScratchPool) -> Self {
        let mut builder = Self::new(limits, alignment);
        builder.inner = scratch
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .unwrap_or_default();
        builder.scratch = Some(scratch);
        builder
    }

    pub fn set_buffer_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    pub fn add<T: NoUninit + AnyBitPattern>(
        &mut self,
        value: &[T],
    ) -> Result<Range<wgpu::BufferAddress>, BufferFull> {
        let start_pos = if !self.inner.is_empty() {
            if self.align_mask > 0 {
                // Pad the internal buffer to match alignment requirements
                // Pad on creation so that we don't wastefully pad the end of the buffer
                (self.inner.len() + self.align_mask) & !self.align_mask
            } else {
                self.inner.len()
            }
        } else {
            0
        };

        let slice = bytemuck::cast_slice(value);
        if (start_pos + slice.len()) as u64 > self.limit {
            return Err(BufferFull);
        }

        if start_pos > 0 && self.align_mask > 0 {
            self.inner.resize(start_pos, 0);
        }

        self.inner.extend_from_slice(slice);
        Ok((start_pos as wgpu::BufferAddress)..(self.inner.len() as wgpu::BufferAddress))
    }

    pub fn finish(
        self,
        device: &crate::gpu_resources::Device,
        label: Option<String>,
        usage: wgpu::BufferUsages,
    ) -> Result<crate::gpu_resources::Buffer, crate::gpu_runtime::RenderFailure> {
        Ok(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: label.as_deref(),
                contents: &self.inner,
                usage,
            })?,
        )
    }

    pub fn copy_to_at(
        self,
        staging_belt: &mut crate::gpu_resources::StagingBelt,
        encoder: &mut crate::gpu_resources::CommandEncoder,
        buffer: &crate::gpu_resources::Buffer,
        offset: u64,
    ) -> Result<(), crate::gpu_runtime::RenderFailure> {
        let capacity = staging_belt.page_size() as usize;
        for (index, chunk) in self.inner.chunks(capacity).enumerate() {
            encoder.flush_uploads_if_needed(capacity as u64)?;
            if let Some(length) = wgpu::BufferSize::new(chunk.len() as u64) {
                staging_belt
                    .write_buffer(encoder, buffer, offset + (index * capacity) as u64, length)?
                    .copy_from_slice(chunk);
            }
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_scratch_reuses_storage_without_reusing_contents_and_caps_retention() {
        let scratch = ScratchPool::default();
        let limits = wgpu::Limits::default();
        let mut first = BufferBuilder::with_scratch(&limits, 0, scratch.clone());
        first.add(&vec![42u8; 1024]).unwrap();
        let pointer = first.inner.as_ptr();
        drop(first);
        let mut next = BufferBuilder::with_scratch(&limits, 0, scratch.clone());
        assert_eq!(next.len(), 0);
        assert_eq!(next.inner.as_ptr(), pointer);
        next.add(&[7u32; 4]).unwrap();
        assert_eq!(next.len(), 16);
        assert_eq!(&next.inner[..4], &7u32.to_ne_bytes());
        drop(next);
        let mut builders = Vec::new();
        for _ in 0..16 {
            let mut builder = BufferBuilder::with_scratch(&limits, 0, scratch.clone());
            builder.add(&vec![0u8; 512 * 1024]).unwrap();
            builders.push(builder);
        }
        drop(builders);
        let buffers = scratch.0.lock().unwrap();
        assert!(buffers.len() <= 8);
        assert!(buffers.iter().map(Vec::capacity).sum::<usize>() <= 2 * 1024 * 1024);
    }
}
