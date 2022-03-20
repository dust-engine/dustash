use std::{mem::ManuallyDrop, ops::Deref, sync::Arc};

use crate::{
    resources::{alloc::MemBuffer, buffer::vec_discrete::VecDiscrete},
    Device,
};
use ash::extensions::khr;
use ash::vk;
pub mod build;

pub struct AccelerationStructureLoader {
    device: Arc<Device>,
    loader: khr::AccelerationStructure,
}
impl AccelerationStructureLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::AccelerationStructure::new(device.instance(), &device);
        Self { device, loader }
    }
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl Deref for AccelerationStructureLoader {
    type Target = khr::AccelerationStructure;

    fn deref(&self) -> &Self::Target {
        &self.loader
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
    primitives_buffer: Option<VecDiscrete<vk::AabbPositionsKHR>>,
    backing_buffer: ManuallyDrop<MemBuffer>,
    compacted: bool,

    ty: vk::AccelerationStructureTypeKHR,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    num_primitives: u64,
    geometries_num_primitives: Vec<u32>,
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_acceleration_structure(self.raw, None);
            ManuallyDrop::drop(&mut self.backing_buffer);
        }
    }
}
