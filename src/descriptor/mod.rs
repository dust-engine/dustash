mod layout;
mod pool;
mod vec;

use std::sync::{Arc, Mutex};

use crate::{Device, HasDevice};

use ash::{prelude::VkResult, vk};

pub use layout::DescriptorSetLayout;
pub use pool::DescriptorPool;
pub use vec::{DescriptorVec, DescriptorVecBinding};

pub struct DescriptorSet {
    raw: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
}
impl DescriptorSet {
    pub fn raw(&self) -> vk::DescriptorSet {
        self.raw
    }
}
impl Drop for DescriptorSet {
    fn drop(&mut self) {
        if !self.pool.free_individual_desc_sets {
            return;
        }
        let raw_pool = self.pool.raw.lock().unwrap();
        tracing::info!(device = ?raw_pool.0, "free descriptor set");
        unsafe {
            (self.pool.device().fp_v1_0().free_descriptor_sets)(
                self.pool.device().handle(),
                raw_pool.0,
                1,
                &self.raw,
            )
            .result()
            .unwrap();
        }
    }
}
