use ash::{prelude::VkResult, vk};
use std::{ffi::c_void, ops::Deref, sync::Arc};

use crate::{sync::CommandsFuture, Device, HasDevice, MemoryHeap, MemoryType};

use super::buffer::HasBuffer;

pub use vk::BufferUsageFlags;
pub use vk_mem::{Alloc, Allocation, AllocationCreateFlags, MemoryUsage};

pub struct Allocator {
    pub(crate) allocator: vk_mem::Allocator,
    device: Arc<Device>,
    memory_model: DeviceMemoryModel,
    pub(crate) heaps: Box<[MemoryHeap]>,
    pub(crate) types: Box<[MemoryType]>,
}
impl crate::HasDevice for Allocator {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

pub enum DeviceMemoryModel {
    Integrated,
    /// Discrete GPU without HOST_VISIBLE DEVICE_LOCAL memory
    Discrete,
    DiscreteBar,
    DiscreteReBar,
}

impl Allocator {
    pub fn new(device: Arc<Device>) -> Self {
        let mut allocator_flags: vk_mem::AllocatorCreateFlags =
            vk_mem::AllocatorCreateFlags::empty();
        if device
            .physical_device()
            .features()
            .v12
            .buffer_device_address
            == vk::TRUE
        {
            allocator_flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        }

        let allocator = vk_mem::Allocator::new(
            vk_mem::AllocatorCreateInfo::new(
                device.instance().as_ref().deref(),
                device.as_ref().deref(),
                device.physical_device().raw(),
            )
            .vulkan_api_version(vk::make_api_version(0, 1, 3, 0))
            .flags(allocator_flags),
        )
        .unwrap();
        let (heaps, types) = device.physical_device().get_memory_properties();
        let memory_model = if device.physical_device().integrated() {
            DeviceMemoryModel::Integrated
        } else {
            let bar_heap = types
                .iter()
                .find(|ty| {
                    ty.property_flags.contains(
                        vk::MemoryPropertyFlags::DEVICE_LOCAL
                            | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    ) && heaps[ty.heap_index as usize]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|a| &heaps[a.heap_index as usize]);
            if let Some(bar_heap) = bar_heap {
                if bar_heap.size <= 256 * 1024 * 1024 {
                    // regular 256MB bar
                    DeviceMemoryModel::DiscreteBar
                } else {
                    DeviceMemoryModel::DiscreteReBar
                }
            } else {
                // Can't find a BAR heap
                DeviceMemoryModel::Discrete
            }
        };
        Self {
            allocator,
            device,
            heaps,
            types,
            memory_model,
        }
    }

    pub(crate) fn create_info_by_scenario(
        &self,
        flags: vk_mem::AllocationCreateFlags,
        scenario: &MemoryAllocScenario,
    ) -> vk_mem::AllocationCreateInfo {
        let mut required_flags = vk::MemoryPropertyFlags::empty();
        let mut preferred_flags = vk::MemoryPropertyFlags::empty();
        let mut non_preferred_flags = vk::MemoryPropertyFlags::empty();
        let mut memory_usage = vk_mem::MemoryUsage::Unknown;
        match scenario {
            MemoryAllocScenario::StagingBuffer => {
                required_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated => {
                        preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                    }
                    DeviceMemoryModel::Discrete
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        non_preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                    }
                }
            }
            MemoryAllocScenario::DeviceAccess => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
            }
            MemoryAllocScenario::AssetBuffer => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated => {
                        preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                    DeviceMemoryModel::Discrete
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        non_preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                }
            }
            MemoryAllocScenario::DynamicUniform => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                    _ => {}
                }
            }
            MemoryAllocScenario::DynamicStorage => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                    _ => {}
                }
            }
            MemoryAllocScenario::Custom {
                memory_usage: memory_usage_self,
                require_flags: required_flags_self,
                preferred_flags: preferred_flags_self,
                non_preferred_flags: non_preferred_flags_self,
            } => {
                memory_usage = *memory_usage_self;
                required_flags = *required_flags_self;
                preferred_flags = *preferred_flags_self;
                non_preferred_flags = *non_preferred_flags_self;
            }
        }
        non_preferred_flags |= vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD;
        vk_mem::AllocationCreateInfo {
            flags,
            usage: memory_usage,
            required_flags,
            preferred_flags,
            ..Default::default()
        }
    }

    pub fn allocate_memory(
        &self,
        memory_requirements: &ash::vk::MemoryRequirements,
        create_info: &vk_mem::AllocationCreateInfo,
    ) -> VkResult<Allocation> {
        let allocation = unsafe {
            self.allocator
                .allocate_memory(memory_requirements, create_info)
        }?;
        Ok(allocation)
    }

    pub fn allocate_buffer(self: &Arc<Self>, request: &BufferRequest) -> VkResult<MemBuffer> {
        let build_info = vk::BufferCreateInfo::builder()
            .size(request.size)
            .usage(request.usage)
            .sharing_mode(request.sharing_mode)
            .queue_family_indices(request.queue_families)
            .build();
        let create_info = self.create_info_by_scenario(request.allocation_flags, &request.scenario);
        let (buffer, allocation) = unsafe {
            if request.alignment == 0 {
                self.allocator.create_buffer(&build_info, &create_info)
            } else {
                self.allocator.create_buffer_with_alignment(
                    &build_info,
                    &create_info,
                    request.alignment,
                )
            }
        }?;
        let (memory_flags, ptr) = unsafe {
            let allocation_info = self.allocator.get_allocation_info(&allocation).unwrap();
            let memory_flags = self.types[allocation_info.memory_type as usize].property_flags;
            let ptr = allocation_info.mapped_data;
            (memory_flags, ptr)
        };
        let device_address = if request
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        {
            unsafe {
                self.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::builder()
                        .buffer(buffer)
                        .build(),
                )
            }
        } else {
            0
        };
        Ok(MemBuffer {
            allocator: self.clone(),
            buffer,
            memory: allocation,
            size: request.size,
            alignment: request.alignment,
            memory_flags,
            ptr,
            device_address,
        })
    }

    pub fn allocate_buffer_with_data(
        self: &Arc<Self>,
        request: BufferRequest,
        f: impl FnOnce(&mut [u8]),
        commands_future: &mut CommandsFuture,
    ) -> VkResult<Arc<MemBuffer>> {
        match &request.scenario {
            MemoryAllocScenario::DeviceAccess => {
                panic!("Unable to allocate Device-only buffer with data")
            }
            _ => {}
        }
        let request = BufferRequest {
            size: request.size,
            alignment: request.alignment,
            usage: request.usage | vk::BufferUsageFlags::TRANSFER_DST,
            scenario: request.scenario,
            ..Default::default()
        };

        let mut buffer = self.allocate_buffer(&request)?;
        if buffer
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            buffer.map_scoped(f);
            return Ok(Arc::new(buffer));
        }
        let mut staging_buffer = self.allocate_buffer(&BufferRequest {
            size: request.size,
            alignment: request.alignment,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            scenario: MemoryAllocScenario::StagingBuffer,
            ..Default::default()
        })?;
        staging_buffer.map_scoped(f);

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
pub enum MemoryAllocScenario {
    /// On integrated GPU, allocate buffer on DEVICE_LOCAL, HOST_VISIBLE non-cached memory.
    /// If no such memory exist, allocate buffer on HOST_VISIBLE non-cached memory.
    ///
    /// On discrete and SAM GPU, allocate buffer on HOST_VISIBLE, non-DEVICE_LOCAL memory.
    StagingBuffer,
    /// On all GPUs, allocate on DEVICE_LOCAL, non-HOST_VISIBLE memory.
    DeviceAccess,
    /// For uploading large assets. Always use staging buffer on discrete GPUs but prefer
    /// HOST_VISIBLE on integrated GPUs.
    AssetBuffer,
    /// For small uniform buffers that frequently gets updated
    /// On integrated GPU, allocate buffer on DEVICE_LOCAL, HOST_VISIBLE, HOST_CACHED memory.
    /// On discrete GPU, allocate buffer on BAR (DEVICE_LOCAL, HOST_VISIBLE, non-cached) if possible.
    /// Otherwise, explicit transfer is required and the buffer will be allocated on DEVICE_LOCAL.
    /// On discrete GPU with SAM, allocate buffer on BAR.
    DynamicUniform,
    /// For large storage buffers that frequently gets updated on certain parts.
    /// On integrated GPU, allocate buffer on DEVICE_LOCAL, HOST_VISIBLE, HOST_CACHED memory.
    /// On discrete GPU, allocate buffer on DEVICE_LOCAL, non host-visible memory.
    /// On discrete GPU with SAM, allocate buffer on BAR.
    DynamicStorage,
    Custom {
        memory_usage: vk_mem::MemoryUsage,
        require_flags: vk::MemoryPropertyFlags,
        preferred_flags: vk::MemoryPropertyFlags,
        non_preferred_flags: vk::MemoryPropertyFlags,
    },
}
#[derive(Clone)]
pub struct BufferRequest<'a> {
    pub size: u64,
    /// If this value is 0, the memory will be allocated based on the buffer requirements.
    /// The actual alignment used on the allocation is buffer_request.alignment.max(buffer_requirements.alignment).
    pub alignment: u64,
    pub usage: vk::BufferUsageFlags,
    pub scenario: MemoryAllocScenario,
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
            scenario: MemoryAllocScenario::DeviceAccess,
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
    pub memory_flags: vk::MemoryPropertyFlags,
    device_address: vk::DeviceAddress,
    pub ptr: *mut c_void,
}
unsafe impl Send for MemBuffer {}
unsafe impl Sync for MemBuffer {}

impl HasBuffer for MemBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

impl HasDevice for MemBuffer {
    fn device(&self) -> &Arc<Device> {
        self.allocator.device()
    }
}

impl crate::debug::DebugObject for MemBuffer {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::BUFFER;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.buffer) }
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

    pub fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.memory_flags
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

    pub fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.allocator.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder()
                    .buffer(self.buffer)
                    .build(),
            )
        }
    }

    pub fn device_local(&self) -> bool {
        self.memory_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }
    pub fn host_visible(&self) -> bool {
        self.memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    pub fn make_device_local(self: Arc<Self>, commands_future: &mut crate::sync::CommandsFuture) -> Arc<Self> {
        if self.device_local() {
            return self
        }
        let size = self.size();

        let buffer = self.allocator.allocate_buffer(&BufferRequest {
            size,
            alignment: self.alignment(),
            usage: ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | ash::vk::BufferUsageFlags::TRANSFER_DST,
            scenario: MemoryAllocScenario::DeviceAccess,
            allocation_flags: AllocationCreateFlags::MAPPED,
            ..Default::default()
        }).unwrap();
        let buffer = Arc::new(buffer);

        commands_future.then_commands(|mut recorder| {
            recorder.copy_buffer(self, buffer.clone(), &[
                ash::vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }
            ]);
        });
        buffer
    }

    pub fn map_scoped(&mut self, f: impl FnOnce(&mut [u8]) -> ()) {
        unsafe {
            let ptr = if self.ptr.is_null() {
                self.allocator
                    .allocator
                    .map_memory(&mut self.memory)
                    .unwrap()
            } else {
                self.ptr as *mut u8
            };
            let slice = std::slice::from_raw_parts_mut(ptr, self.size as usize);
            f(slice);
            self.allocator.allocator.unmap_memory(&mut self.memory);
        }
    }

    pub fn get_mut(&self) -> &mut [u8] {
        assert!(!self.ptr.is_null());
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.size as usize)
        }
    }
}
