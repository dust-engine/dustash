use std::{mem::ManuallyDrop, ops::Deref, sync::Arc};

use crate::{
    resources::{alloc::MemBuffer, buffer::vec_discrete::VecDiscrete},
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

impl AccelerationStructure {
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
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
