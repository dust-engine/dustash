use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use crate::Device;

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    pub(super) raw: vk::DescriptorSetLayout,
}
impl DescriptorSetLayout {
    pub unsafe fn new(device: Arc<Device>, info: &vk::DescriptorSetLayoutCreateInfo) -> VkResult<Self> {
        let raw = device.create_descriptor_set_layout(info, None)?;
        Ok(Self { device, raw })
    }
    pub fn raw(&self) -> vk::DescriptorSetLayout {
        self.raw
    }
}
impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        tracing::info!(device = ?self.raw, "destroy descriptor layout");
        unsafe {
            self.device.destroy_descriptor_set_layout(self.raw, None);
        }
    }
}
