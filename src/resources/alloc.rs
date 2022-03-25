use ash::vk;
use std::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{Arc, RwLock},
};

use crate::{command::recorder::CommandRecorder, Device};

use super::buffer::HasBuffer;

pub enum StagingStrategy {
    /// There does not exist any DeviceLocal, HostVisible memory.
    /// e.g. NVIDIA
    Required,

    /// There is a small DeviceLocal, HostVisible memory pool. Most data should be moved to the device local memory,
    /// except frequently updated small data.
    BarAccess,

    /// There is a large DeviceLocal, HostVisible memory pool. All data could do host writes, except
    SamAccess,

    /// Most data should not be moved, except small, fast but never updated data.
    Avoid,

    /// Should never use a staging buffer.
    Never,
}
impl StagingStrategy {
    pub fn from_memory_properties(
        heaps: &[crate::physical_device::MemoryHeap],
        types: &[crate::physical_device::MemoryType],
        device_type: vk::PhysicalDeviceType,
    ) -> Self {
        let max_device_local_host_visible_type = types
            .iter()
            .filter(|&ty| {
                ty.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
            })
            .max_by_key(|&ty| heaps[ty.heap_index as usize].size);

        let max_device_local_only_type = types
            .iter()
            .filter(|&ty| {
                ty.property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && !ty
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
            .max_by_key(|&ty| heaps[ty.heap_index as usize].size);

        if device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            || device_type == vk::PhysicalDeviceType::CPU
        {
            // Integrated GPU.
            if max_device_local_only_type.is_some() {
                StagingStrategy::Avoid
            } else {
                StagingStrategy::Never
            }
        } else {
            if let Some(max_device_local_host_visible) = max_device_local_host_visible_type {
                if heaps[max_device_local_host_visible.heap_index as usize].size < 512 * 1024 * 1024
                {
                    // Small device local host visible.
                    StagingStrategy::BarAccess
                } else {
                    // Large device local host visible.
                    StagingStrategy::SamAccess
                }
            } else {
                // No device local host visible buffer.
                StagingStrategy::Required
            }
        }
    }
}

type GpuAllocator = gpu_alloc::GpuAllocator<vk::DeviceMemory>;
pub struct Allocator {
    allocator: RwLock<GpuAllocator>,
    device: Arc<Device>,
    staging_strategy: StagingStrategy,
}
impl Allocator {
    pub fn new(device: Arc<Device>) -> Self {
        use gpu_alloc::{Config, DeviceProperties, MemoryHeap, MemoryType};
        use gpu_alloc_ash::memory_properties_from_ash;
        use std::borrow::Cow;

        let (heaps, types) = device.physical_device().get_memory_properties();
        let strategy = StagingStrategy::from_memory_properties(
            &heaps,
            &types,
            device.physical_device().properties().device_type,
        );

        let config = Config::i_am_prototyping();

        let props = DeviceProperties {
            memory_types: Cow::Owned(
                types
                    .iter()
                    .map(|memory_type| MemoryType {
                        props: memory_properties_from_ash(memory_type.property_flags),
                        heap: memory_type.heap_index,
                    })
                    .collect(),
            ),
            memory_heaps: Cow::Owned(
                heaps
                    .iter()
                    .map(|memory_heap| MemoryHeap {
                        size: memory_heap.size,
                    })
                    .collect(),
            ),
            max_memory_allocation_count: device
                .physical_device()
                .properties()
                .limits
                .max_memory_allocation_count,
            max_memory_allocation_size: device
                .physical_device()
                .properties()
                .v11
                .max_memory_allocation_size,
            non_coherent_atom_size: device
                .physical_device()
                .properties()
                .limits
                .non_coherent_atom_size,
            buffer_device_address: device
                .physical_device()
                .features()
                .v12
                .buffer_device_address
                != 0,
        };
        let allocator: GpuAllocator = GpuAllocator::new(config, props);

        Self {
            allocator: RwLock::new(allocator),
            device: device.clone(),
            staging_strategy: strategy,
        }
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn allocate_memory(
        &self,
        request: gpu_alloc::Request,
    ) -> Result<MemoryBlock, gpu_alloc::AllocationError> {
        unsafe {
            let block = self
                .allocator
                .write()
                .unwrap()
                .alloc(gpu_alloc_ash::AshMemoryDevice::wrap(&self.device), request)?;
            Ok(MemoryBlock(block))
        }
    }

    pub fn allocate_buffer(
        self: &Arc<Self>,
        request: BufferRequest,
    ) -> Result<MemBuffer, gpu_alloc::AllocationError> {
        unsafe {
            use gpu_alloc::{AllocationError, Request};
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(request.size)
                        .usage(request.usage)
                        .sharing_mode(request.sharing_mode)
                        .queue_family_indices(request.queue_families)
                        .build(),
                    None,
                )
                .map_err(|err| match err {
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => AllocationError::OutOfDeviceMemory,
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY => AllocationError::OutOfHostMemory,
                    _ => panic!(),
                })?;

            let reqs = self.device.get_buffer_memory_requirements(buffer);

            let mem = self.allocator.write().unwrap().alloc(
                gpu_alloc_ash::AshMemoryDevice::wrap(&self.device),
                Request {
                    size: reqs.size,
                    align_mask: reqs.alignment.max(request.alignment) - 1,
                    usage: request.memory_usage,
                    memory_types: reqs.memory_type_bits,
                },
            )?;

            self.device
                .bind_buffer_memory(buffer, *mem.memory(), mem.offset())
                .map_err(|err| match err {
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => AllocationError::OutOfDeviceMemory,
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY => AllocationError::OutOfHostMemory,
                    _ => panic!(),
                })?;

            Ok(MemBuffer {
                memory: MaybeUninit::new(MemoryBlock(mem)),
                buffer,
                allocator: self.clone(),
            })
        }
    }

    // Returns: MainBuffer, StagingBuffer
    pub fn allocate_buffer_with_data(
        self: &Arc<Self>,
        mut request: BufferRequest,
        f: impl FnOnce(&mut [u8]),
        command_recorder: &mut CommandRecorder,
    ) -> Result<Arc<MemBuffer>, gpu_alloc::AllocationError> {
        let should_staging = match self.staging_strategy {
            StagingStrategy::Never | StagingStrategy::Avoid => false,
            _ => true,
        };
        if should_staging {
            request.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let mut target_buf = self.allocate_buffer(request.clone())?;
        if !should_staging {
            target_buf.map_scoped(0, request.size as usize, f);
            return Ok(Arc::new(target_buf));
        }

        let mut staging_buf = self.allocate_buffer(BufferRequest {
            size: request.size,
            alignment: request.alignment,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_usage: gpu_alloc::UsageFlags::UPLOAD | gpu_alloc::UsageFlags::TRANSIENT,
            ..Default::default()
        })?;
        staging_buf.map_scoped(0, request.size as usize, f);

        let target_buf = Arc::new(target_buf);
        command_recorder.copy_buffer(
            staging_buf,
            target_buf.clone(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: request.size,
            }],
        );
        Ok(target_buf)
    }
}
pub struct MemoryBlock(gpu_alloc::MemoryBlock<vk::DeviceMemory>);
impl Deref for MemoryBlock {
    type Target = gpu_alloc::MemoryBlock<vk::DeviceMemory>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MemoryBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl MemoryBlock {
    /// Writes into host-visible memory
    pub unsafe fn write_bytes(
        &mut self,
        device: &ash::Device,
        offset: u64,
        data: &[u8],
    ) -> Result<(), gpu_alloc::MapError> {
        assert!(self
            .0
            .props()
            .intersects(gpu_alloc::MemoryPropertyFlags::HOST_VISIBLE));
        self.0
            .write_bytes(gpu_alloc_ash::AshMemoryDevice::wrap(device), offset, data)
    }

    pub unsafe fn bind_buffer(
        &self,
        device: &ash::Device,
        buffer: vk::Buffer,
    ) -> Result<(), gpu_alloc::AllocationError> {
        use gpu_alloc::AllocationError;
        device
            .bind_buffer_memory(buffer, *self.0.memory(), self.0.offset())
            .map_err(|err| match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => AllocationError::OutOfDeviceMemory,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => AllocationError::OutOfHostMemory,
                _ => panic!(),
            })?;
        Ok(())
    }

    pub unsafe fn map(
        &mut self,
        device: &ash::Device,
        offset: u64,
        size: usize,
    ) -> Result<NonNull<u8>, gpu_alloc::MapError> {
        self.0
            .map(gpu_alloc_ash::AshMemoryDevice::wrap(device), offset, size)
    }
    pub unsafe fn unmap(&mut self, device: &ash::Device) -> bool {
        self.0.unmap(gpu_alloc_ash::AshMemoryDevice::wrap(device))
    }
}

#[derive(Clone)]
pub struct BufferRequest<'a> {
    pub size: u64,
    /// If this value is 0, the memory will be allocated based on the buffer requirements.
    /// The actual alignment used on the allocation is buffer_request.alignment.max(buffer_requirements.alignment).
    pub alignment: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_usage: gpu_alloc::UsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub queue_families: &'a [u32],
}
impl<'a> Default for BufferRequest<'a> {
    fn default() -> Self {
        Self {
            size: 0,
            alignment: 0,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: gpu_alloc::UsageFlags::empty(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_families: &[],
        }
    }
}

pub struct MemBuffer {
    allocator: Arc<Allocator>,
    pub buffer: vk::Buffer,
    pub memory: MaybeUninit<MemoryBlock>,
}

impl HasBuffer for MemBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for MemBuffer {
    fn drop(&mut self) {
        unsafe {
            let device = gpu_alloc_ash::AshMemoryDevice::wrap(&self.allocator.device);

            let memory = std::mem::replace(&mut self.memory, MaybeUninit::uninit());
            let memory = memory.assume_init();
            let mut allocator = self.allocator.allocator.write().unwrap();
            allocator.dealloc(device, memory.0);
            drop(allocator);
            self.allocator.device.destroy_buffer(self.buffer, None);
        }
    }
}

impl MemBuffer {
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

    pub fn memory(&self) -> &MemoryBlock {
        unsafe { self.memory.assume_init_ref() }
    }

    pub fn memory_mut(&mut self) -> &mut MemoryBlock {
        unsafe { self.memory.assume_init_mut() }
    }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.allocator.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder()
                    .buffer(self.buffer)
                    .build(),
            )
        }
    }
    pub fn write_bytes(&mut self, offset: u64, data: &[u8]) {
        unsafe {
            self.memory
                .assume_init_mut()
                .write_bytes(&self.allocator.device, offset, data)
                .unwrap();
        }
    }
    pub fn map_scoped(&mut self, offset: u64, size: usize, f: impl FnOnce(&mut [u8])) {
        unsafe {
            let ptr = self
                .memory
                .assume_init_mut()
                .map(&self.allocator.device, offset, size)
                .unwrap();
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
            f(slice);
            self.memory.assume_init_mut().unmap(&self.allocator.device);
        }
    }
}
