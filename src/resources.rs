use std::sync::Arc;

use ash::extensions::*;
use ash::vk;

pub struct Buffer {
    device: Arc<ash::Device>,
    pub(crate) buffer: vk::Buffer,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { self.device.destroy_buffer(self.buffer, None) }
    }
}

pub trait HasImage {
    fn raw_image(&self) -> vk::Image;
}

impl HasImage for vk::Image {
    fn raw_image(&self) -> vk::Image {
        *self
    }
}

pub struct Image {
    device: Arc<ash::Device>,
    pub(crate) image: vk::Image,
}

impl HasImage for Image {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { self.device.destroy_image(self.image, None) }
    }
}

pub struct AccelerationStructure {
    loader: Arc<khr::AccelerationStructure>,
    accel_struct: vk::AccelerationStructureKHR,
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_acceleration_structure(self.accel_struct, None);
        }
    }
}
