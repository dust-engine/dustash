use ash::{vk::{self}, prelude::VkResult};
use std::{
    ops::{Deref},
    sync::{Arc},
};

use crate::{sync::CommandsFuture, Device};

use super::buffer::HasBuffer;

pub use vk::BufferUsageFlags;
use vk_mem::{Alloc, Allocation, MemoryUsage, AllocationCreateFlags};

pub struct Allocator {
    allocator: vk_mem::Allocator,
    device: Arc<Device>,
}
impl crate::HasDevice for Allocator {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Allocator {
    pub fn new(device: Arc<Device>) -> Self {
        let allocator = vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
            device.instance().as_ref().deref(),
            device.as_ref().deref(),
            device.physical_device().raw()
        )).unwrap();
        Self {
            allocator,
            device,
        }
    }

    pub fn allocate_memory(
        &self,
        memory_requirements: &ash::vk::MemoryRequirements,
        create_info: &vk_mem::AllocationCreateInfo,
    ) -> VkResult<Allocation> {
        let allocation = unsafe {
            self.allocator.allocate_memory(memory_requirements, create_info)
        }?;
        Ok(allocation)
    }

    pub fn allocate_buffer(
        self: &Arc<Self>,
        request: &BufferRequest,
    ) -> VkResult<MemBuffer> {
        let build_info = vk::BufferCreateInfo::builder()
        .size(request.size)
        .usage(request.usage)
        .sharing_mode(request.sharing_mode)
        .queue_family_indices(request.queue_families)
        .build();
        let create_info = vk_mem::AllocationCreateInfo {
            flags: request.allocation_flags,
            usage: request.memory_usage,
            required_flags: request.memory_required_flags,
            preferred_flags: request.memory_preferred_flags,
            ..Default::default()
        };
        let (buffer, allocation) = unsafe {
            if request.alignment == 0 {
                self.allocator.create_buffer(&build_info, &create_info)
            } else {
                self.allocator.create_buffer_with_alignment(&build_info, &create_info, request.alignment)
            }
        }?;
        Ok(MemBuffer {
            allocator: self.clone(),
            buffer,
            memory: allocation,
            size: request.size,
            alignment: request.alignment
        })
    }

    // Returns: MainBuffer, StagingBuffer
    /// On integrated GPUs, this should allocate device-local buffer with host-visible.
    /// On discrete GPUs with no BAR, this should allocate device-local buffer without HOST_VISIBLE.
    /// On discrete GPUs with 256MB BAR, this should allocate device-local buffer without HOST_VISIBLE.
    /// On discrete GPUs with reBAR, this should still allocate device-local buffer without HOST_VISIBLE.
    /// For large one-off transfers like this, vkCmdCopyBuffer is still likely faster.
    pub fn allocate_buffer_with_data(
        self: &Arc<Self>,
        request: BufferRequest,
        f: impl FnOnce(&mut [u8]),
        commands_future: &mut CommandsFuture,
    ) -> VkResult<Arc<MemBuffer>> {
        let mut request = BufferRequest {
            size: request.size,
            alignment: request.alignment,
            usage: request.usage | vk::BufferUsageFlags::TRANSFER_DST,
            memory_usage: MemoryUsage::Unknown,
            allocation_flags: AllocationCreateFlags::MAPPED,
            memory_preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };
        if self.device.physical_device().integrated() {
            request.memory_preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
        }
        // TODO: declare HOST_VISIBLE as undesired for discrete graphics

        let mut buffer = self.allocate_buffer(&request)?;
        let info = unsafe {
            self.allocator.get_allocation_info(&buffer.memory).unwrap()
        };
        if !info.mapped_data.is_null() {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(info.mapped_data as *mut u8, request.size as usize);
                f(slice);
                self.allocator.unmap_memory(&mut buffer.memory);
            };
            return Ok(Arc::new(buffer))
        }
        let mut staging_buffer = self.allocate_buffer(&BufferRequest {
            size: request.size,
            alignment: request.alignment,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_usage: MemoryUsage::Auto,
            allocation_flags: AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        })?;
        let staging_info = unsafe {
            self.allocator.get_allocation_info(&staging_buffer.memory).unwrap()
        };
        assert!(!staging_info.mapped_data.is_null());
        unsafe {
            let slice = std::slice::from_raw_parts_mut(staging_info.mapped_data as *mut u8, request.size as usize);
            f(slice);
            self.allocator.unmap_memory(&mut staging_buffer.memory);
        }

        let target_buf = Arc::new(buffer);
        commands_future.then_commands(|mut recorder| {
            recorder.copy_buffer(
                staging_buffer,
                target_buf.clone(), 
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: request.size,
                }],
            );
        });
        Ok(target_buf)
    }
}
#[derive(Clone)]
pub struct BufferRequest<'a> {
    pub size: u64,
    /// If this value is 0, the memory will be allocated based on the buffer requirements.
    /// The actual alignment used on the allocation is buffer_request.alignment.max(buffer_requirements.alignment).
    pub alignment: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_usage: MemoryUsage,
    pub memory_required_flags: vk::MemoryPropertyFlags,
    pub memory_preferred_flags: vk::MemoryPropertyFlags,
    pub allocation_flags: AllocationCreateFlags,
    pub sharing_mode: vk::SharingMode,
    pub queue_families: &'a [u32],
}
impl<'a> Default for BufferRequest<'a> {
    fn default() -> Self {
        Self {
            size: 0,
            alignment: 0,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: vk_mem::MemoryUsage::Auto,
            memory_required_flags: vk::MemoryPropertyFlags::empty(),
            memory_preferred_flags: vk::MemoryPropertyFlags::empty(),
            allocation_flags: vk_mem::AllocationCreateFlags::empty(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_families: &[],
        }
    }
}

pub struct MemBuffer {
    allocator: Arc<Allocator>,
    pub buffer: vk::Buffer,
    pub memory: Allocation,
    size: u64,
    alignment: u64,
}

impl HasBuffer for MemBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for MemBuffer {
    fn drop(&mut self) {
        unsafe {
            let mut memory: Allocation = std::mem::zeroed();
            std::mem::swap(&mut memory, &mut self.memory);
            self.allocator.allocator.destroy_buffer(self.buffer, memory)
        }
    }
}

impl MemBuffer {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    /*
    pub fn set_debug_name_raw(&self, device: &ash::Device, debug_utils: &DebugUtils, name: &CStr) {
        unsafe {
            debug_utils
                .debug_utils_set_object_name(
                    device.handle(),
                    &vk::DebugUtilsObjectNameInfoEXT::builder()
                        .object_type(vk::ObjectType::BUFFER)
                        .object_name(name)
                        .object_handle(std::mem::transmute(self.buffer))
                        .build(),
                )
                .unwrap();
        }
    }
    */

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.allocator.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder()
                    .buffer(self.buffer)
                    .build(),
            )
        }
    }

    pub fn map_scoped(&mut self, f: impl FnOnce(&mut [u8]) -> ()) {
        unsafe {
            let ptr = self.allocator.allocator.map_memory(&mut self.memory).unwrap();
            let slice = std::slice::from_raw_parts_mut(ptr, self.size as usize);
            f(slice);
            self.allocator.allocator.unmap_memory(&mut self.memory);
        }
    }
}
