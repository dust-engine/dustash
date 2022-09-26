use ash::{prelude::VkResult, vk};
use std::sync::Arc;

use crate::{command::recorder::CommandBufferResource, Device};
//pub mod vec_discrete;

pub struct Buffer {
    device: Arc<Device>,
    pub(crate) raw: vk::Buffer,
}

impl Buffer {
    pub fn new(device: Arc<Device>, create_info: &vk::BufferCreateInfo) -> VkResult<Self> {
        let buffer = unsafe { device.create_buffer(create_info, None)? };
        Ok(Self {
            device,
            raw: buffer,
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        tracing::debug!(buffer = ?self.raw, "drop buffer");
        unsafe { self.device.destroy_buffer(self.raw, None) }
    }
}

pub trait HasBuffer: Send + Sync + 'static {
    fn raw_buffer(&self) -> vk::Buffer;
    fn boxed_type_erased(self: Box<Self>) -> Box<dyn Send + Sync> {
        Box::new(self)
    }
}

impl HasBuffer for vk::Buffer {
    fn raw_buffer(&self) -> vk::Buffer {
        *self
    }
}

impl<T: HasBuffer> HasBuffer for Arc<T> {
    fn raw_buffer(&self) -> vk::Buffer {
        let r: &T = self.as_ref();
        r.raw_buffer()
    }
}
