use std::sync::{Arc, Mutex};

use crate::Device;

use ash::{prelude::VkResult, vk};

// Make pool non-clone, non-copy
struct DescriptorPoolInner(vk::DescriptorPool);

pub struct DescriptorPool {
    device: Arc<Device>,
    raw: Mutex<DescriptorPoolInner>,
}

impl DescriptorPool {
    pub fn new(
        device: Arc<Device>,
        info: &vk::DescriptorPoolCreateInfo,
    ) -> VkResult<DescriptorPool> {
        let raw = unsafe { device.create_descriptor_pool(info, None)? };
        Ok(DescriptorPool {
            device,
            raw: Mutex::new(DescriptorPoolInner(raw)),
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

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    raw: vk::DescriptorSetLayout,
}
impl DescriptorSetLayout {
    pub fn new(device: Arc<Device>, info: &vk::DescriptorSetLayoutCreateInfo) -> VkResult<Self> {
        let raw = unsafe { device.create_descriptor_set_layout(info, None)? };
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
        let raw_pool = self.pool.raw.lock().unwrap();
        tracing::info!(device = ?raw_pool.0, "free descriptor set");
        unsafe {
            (self.pool.device.fp_v1_0().free_descriptor_sets)(
                self.pool.device.handle(),
                raw_pool.0,
                1,
                &self.raw,
            )
            .result()
            .unwrap();
        }
    }
}
