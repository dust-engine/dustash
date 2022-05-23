use std::{alloc::Layout, marker::PhantomData, sync::Arc};

use crate::{
    resources::alloc::{Allocator, BufferRequest, MemBuffer},
    sync::CommandsFuture,
};

use super::FrameManager;
use ash::vk;
pub struct PerFrameUniform<T, const PER_IMAGE: bool = false> {
    capacity: u32,
    _marker: PhantomData<T>,
    uniform: Arc<MemBuffer>,
    staging: Option<Arc<MemBuffer>>,
}
unsafe impl<T, const PER_IMAGE: bool> Send for PerFrameUniform<T, PER_IMAGE> {}
unsafe impl<T, const PER_IMAGE: bool> Sync for PerFrameUniform<T, PER_IMAGE> {}

impl<T, const PER_IMAGE: bool> PerFrameUniform<T, PER_IMAGE> {
    pub fn new(frames: &FrameManager, allocator: &Arc<Allocator>) -> Self {
        let num = if PER_IMAGE {
            frames.num_images()
        } else {
            frames.num_frames()
        };

        let layout = Layout::new::<T>().repeat(num).unwrap().0;
        let total_buffer_size = layout.pad_to_align().size();

        let buffer = allocator
            .allocate_buffer(&BufferRequest {
                size: total_buffer_size as u64,
                alignment: layout.align() as u64,
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
                scenario: crate::resources::alloc::MemoryAllocScenario::DynamicUniform,
                allocation_flags: vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            })
            .unwrap();

        let staging_buffer = if !buffer
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            Some(
                allocator
                    .allocate_buffer(&BufferRequest {
                        size: total_buffer_size as u64,
                        alignment: layout.align() as u64,
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        scenario: crate::resources::alloc::MemoryAllocScenario::StagingBuffer,
                        allocation_flags: vk_mem::AllocationCreateFlags::MAPPED,
                        ..Default::default()
                    })
                    .unwrap(),
            )
        } else {
            None
        };
        Self {
            capacity: num as u32,
            _marker: PhantomData,
            uniform: Arc::new(buffer),
            staging: staging_buffer.map(Arc::new),
        }
    }

    pub fn needs_rebuild(&self, frames: &FrameManager) -> bool {
        if PER_IMAGE {
            self.capacity != frames.num_images() as u32
        } else {
            self.capacity != frames.num_frames() as u32
        }
    }

    pub fn write(
        &mut self,
        item: T,
        frame: &super::AcquiredFrame,
        transfer_future: &mut CommandsFuture,
    ) -> (Arc<MemBuffer>, u64) {
        let buffer_to_write = self.staging.as_ref().unwrap_or(&self.uniform);
        let ptr = buffer_to_write.ptr;
        let index = if PER_IMAGE {
            frame.image_index as usize
        } else {
            frame.frame_index
        };

        let layout = Layout::new::<T>().repeat(index).unwrap().0;
        let offset = layout.pad_to_align().size();

        unsafe {
            let ptr: *mut T = ptr.add(offset) as *mut T;
            let ptr = &mut *ptr;
            *ptr = item;
        }

        if let Some(staging_buffer) = self.staging.as_ref() {
            transfer_future.then_commands(|mut recorder| {
                recorder.copy_buffer(
                    staging_buffer.clone(),
                    self.uniform.clone(),
                    &[vk::BufferCopy {
                        src_offset: offset as u64,
                        dst_offset: offset as u64,
                        size: std::mem::size_of::<T>() as u64,
                    }],
                );
            });
        }
        (self.uniform.clone(), offset as u64)
    }
}
