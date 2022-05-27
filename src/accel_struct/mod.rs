use std::{mem::ManuallyDrop, ops::Deref, sync::Arc};

use crate::{
    resources::alloc::{Allocator, BufferRequest, MemBuffer, MemoryAllocScenario},
    sync::CommandsFuture,
    Device, HasDevice,
};
use ash::extensions::khr;
use ash::vk;
pub mod build;

pub struct AccelerationStructureLoader {
    device: Arc<Device>,
    loader: khr::AccelerationStructure,
}

impl crate::HasDevice for AccelerationStructureLoader {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl AccelerationStructureLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::AccelerationStructure::new(device.instance(), &device);
        Self { device, loader }
    }
}
impl Deref for AccelerationStructureLoader {
    type Target = khr::AccelerationStructure;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}

#[derive(Clone, Copy)]
pub enum AccelerationStructureType {
    BottomLevelAABBs,
    BottomLevelTriangles,
    TopLevel,
}

impl Into<vk::GeometryTypeKHR> for AccelerationStructureType {
    fn into(self) -> vk::GeometryTypeKHR {
        use AccelerationStructureType::*;
        match self {
            TopLevel => vk::GeometryTypeKHR::INSTANCES,
            BottomLevelAABBs => vk::GeometryTypeKHR::AABBS,
            BottomLevelTriangles => vk::GeometryTypeKHR::TRIANGLES,
        }
    }
}

impl From<AccelerationStructureType> for vk::AccelerationStructureTypeKHR {
    fn from(ty: AccelerationStructureType) -> Self {
        match ty {
            AccelerationStructureType::BottomLevelAABBs
            | AccelerationStructureType::BottomLevelTriangles => {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            }
            AccelerationStructureType::TopLevel => vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        }
    }
}

// Acceleration structure can be resized and rebuilt
// Buffer data on CPU should be entirely optional.
// can update some entries of the AABB buffer without retransfering all datas.
// can append to the AABB buffer and resize AABB buffer. original data can do GPU to GPU transfers.
// can mark certain entries as DELETED, inactivate them so they don't show up in AS, and reuse them in the future when appending.
// Specialization for uniform memory.
pub struct AccelerationStructure {
    loader: Arc<AccelerationStructureLoader>,
    raw: vk::AccelerationStructureKHR,
    device_address: vk::DeviceAddress,
    backing_buffer: ManuallyDrop<MemBuffer>,
    compacted: bool,

    ty: AccelerationStructureType,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    geometries_num_primitives: Vec<u32>,
}

impl HasDevice for AccelerationStructure {
    fn device(&self) -> &Arc<Device> {
        self.loader.device()
    }
}

impl crate::debug::DebugObject for AccelerationStructure {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::ACCELERATION_STRUCTURE_KHR;
    fn object_handle(&mut self) -> u64 {
        unsafe {
            std::mem::transmute(self.raw)
        }
    }
}

impl AccelerationStructure {
    pub fn new(
        loader: Arc<AccelerationStructureLoader>,
        allocator: &Arc<Allocator>,
        size: vk::DeviceSize,
        ty: AccelerationStructureType,
    ) -> Self {
        let backing_buffer = allocator
            .allocate_buffer(&BufferRequest {
                size,
                alignment: 0,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                scenario: MemoryAllocScenario::DeviceAccess,
                ..Default::default()
            })
            .unwrap();
        let acceleration_structure = unsafe {
            loader
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR {
                        create_flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
                        buffer: backing_buffer.buffer,
                        offset: 0,
                        size,
                        ty: ty.into(),
                        // deviceAddress is the device address requested for the acceleration structure
                        // if the accelerationStructureCaptureReplay feature is being used.
                        device_address: 0,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        let device_address = unsafe {
            loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR {
                    acceleration_structure,
                    ..Default::default()
                },
            )
        };
        Self {
            loader,
            raw: acceleration_structure,
            device_address,
            backing_buffer: ManuallyDrop::new(backing_buffer),
            compacted: false,
            ty,
            flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
            geometries_num_primitives: Vec::new(),
        }
    }
    pub fn make_tlas(
        loader: Arc<AccelerationStructureLoader>,
        allocator: &Arc<Allocator>,
        instances: &[vk::AccelerationStructureInstanceKHR],
        commands_future: &mut CommandsFuture,
    ) -> Arc<AccelerationStructure> {
        let source_buffer_size = std::mem::size_of_val(instances);
        let instances_buffer = allocator
            .allocate_buffer_with_data(
                BufferRequest {
                    size: source_buffer_size as u64,
                    alignment: 0,
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    scenario: MemoryAllocScenario::AssetBuffer,
                    ..Default::default()
                },
                |target_region| {
                    let source_region = unsafe {
                        std::slice::from_raw_parts(
                            instances.as_ptr() as *const u8,
                            source_buffer_size,
                        )
                    };
                    target_region.copy_from_slice(source_region);
                },
                commands_future,
            )
            .unwrap();
        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                    array_of_pointers: vk::FALSE,
                    data: vk::DeviceOrHostAddressConstKHR {
                        device_address: instances_buffer.get_device_address(),
                    },
                    ..Default::default()
                },
            },
            ..Default::default()
        };
        let mut build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            src_acceleration_structure: vk::AccelerationStructureKHR::null(),
            dst_acceleration_structure: vk::AccelerationStructureKHR::null(),
            geometry_count: 1,
            p_geometries: &geometry,
            pp_geometries: std::ptr::null(),
            scratch_data: vk::DeviceOrHostAddressKHR { device_address: 0 },
            ..Default::default()
        };
        let build_sizes = unsafe {
            loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geometry_info,
                &[instances.len() as u32],
            )
        };
        let scratch_buffer = allocator
            .allocate_buffer(&BufferRequest {
                size: build_sizes.build_scratch_size,
                alignment: loader
                    .physical_device()
                    .properties()
                    .acceleration_structure
                    .min_acceleration_structure_scratch_offset_alignment
                    as u64,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                scenario: MemoryAllocScenario::DeviceAccess,
                ..Default::default()
            })
            .unwrap();
        build_geometry_info.scratch_data.device_address = scratch_buffer.get_device_address();

        let mut accel_struct = AccelerationStructure::new(
            loader.clone(),
            &allocator,
            build_sizes.acceleration_structure_size,
            AccelerationStructureType::TopLevel,
        );
        accel_struct.flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
        accel_struct.geometries_num_primitives = vec![instances.len() as u32];
        let accel_struct = Arc::new(accel_struct);
        build_geometry_info.dst_acceleration_structure = accel_struct.raw;
        commands_future.then_commands(|recorder| unsafe {
            loader.cmd_build_acceleration_structures(
                recorder.command_buffer,
                &[build_geometry_info],
                &[&[vk::AccelerationStructureBuildRangeInfoKHR {
                    primitive_count: instances.len() as u32,
                    primitive_offset: 0,
                    first_vertex: 0,
                    transform_offset: 0,
                }]],
            );
            recorder
                .referenced_resources
                .push(Box::new(accel_struct.clone()));
            recorder.referenced_resources.push(Box::new(scratch_buffer));
            recorder
                .referenced_resources
                .push(Box::new(instances_buffer));
        });
        accel_struct
    }
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }
    pub fn raw(&self) -> vk::AccelerationStructureKHR {
        self.raw
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_acceleration_structure(self.raw, None);
            ManuallyDrop::drop(&mut self.backing_buffer);
        }
    }
}
