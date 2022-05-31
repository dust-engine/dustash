use std::sync::{Arc, Mutex};

use ash::{prelude::VkResult, vk};

use crate::{Device, HasDevice};

use super::{DescriptorSet, DescriptorSetLayout};

// Make pool non-clone, non-copy
pub(super) struct DescriptorPoolInner(pub(super) vk::DescriptorPool);

pub struct DescriptorPool {
    device: Arc<Device>,
    pub(super) raw: Mutex<DescriptorPoolInner>,
    pub(super) free_individual_desc_sets: bool,
}

impl HasDevice for DescriptorPool {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl DescriptorPool {
    pub fn new(
        device: Arc<Device>,
        info: &vk::DescriptorPoolCreateInfo,
    ) -> VkResult<DescriptorPool> {
        let free_individual_desc_sets = info
            .flags
            .contains(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let raw = unsafe { device.create_descriptor_pool(info, None)? };
        Ok(DescriptorPool {
            device,
            raw: Mutex::new(DescriptorPoolInner(raw)),
            free_individual_desc_sets,
        })
    }
    pub fn allocate<'a>(
        self: &Arc<Self>,
        layouts: impl IntoIterator<Item = &'a DescriptorSetLayout>,
    ) -> VkResult<Vec<DescriptorSet>> {
        let layouts = layouts.into_iter().map(|l| l.raw).collect::<Vec<_>>();
        unsafe {
            // Safety:
            // - Host access to pAllocateInfo->descriptorPool must be externally synchronized
            // We have &mut self and therefore &mut self.raw
            let raw = self.raw.lock().unwrap();
            let descriptors =
                self.device
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                        descriptor_pool: raw.0,
                        descriptor_set_count: layouts.len() as u32,
                        p_set_layouts: layouts.as_ptr(),
                        ..Default::default()
                    })?;
            drop(raw);

            let descriptors = descriptors
                .into_iter()
                .map(|raw| DescriptorSet {
                    raw,
                    pool: self.clone(),
                })
                .collect();
            Ok(descriptors)
        }
    }
    pub fn free(self: &Arc<Self>, sets: impl IntoIterator<Item = DescriptorSet>) -> VkResult<()> {
        let sets = sets
            .into_iter()
            .inspect(|set| assert!(Arc::ptr_eq(self, &set.pool)))
            .map(|set| {
                let raw = set.raw;
                std::mem::forget(set); // so that the destructor for individual DescriptorSets aren't run.
                raw
            })
            .collect::<Vec<_>>();
        unsafe {
            let raw = self.raw.lock().unwrap();
            self.device.free_descriptor_sets(raw.0, &sets)
        }
    }
}
impl Drop for DescriptorPool {
    fn drop(&mut self) {
        let raw = self.raw.get_mut().unwrap();
        tracing::info!(device = ?raw.0, "drop descriptor pool");
        unsafe {
            self.device.destroy_descriptor_pool(raw.0, None);
        }
    }
}
