use crate::{descriptor::DescriptorSetLayout, Device};
use std::{sync::Arc, collections::HashMap, hash::Hash};
use ash::vk;

use super::pipeline::PipelineLayout;

pub struct PipelineCache {
    device: Arc<Device>,
    descriptor_set_layouts: HashMap<DescriptorSetLayoutCreateInfo, Arc<DescriptorSetLayout>>,
    pipeline_layouts: HashMap<PipelineLayoutCreateInfo, Arc<PipelineLayout>>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Binding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
}
#[derive(Hash, PartialEq, Eq, Clone)]
pub struct DescriptorSetLayoutCreateInfo {
    pub flags: vk::DescriptorSetLayoutCreateFlags,
    /// Sorted by binding index
    pub bindings: Vec<Binding>
}

#[repr(C)]
#[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub struct PushConstantRange {
    pub stage_flags: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct PipelineLayoutCreateInfo {
    pub flags: vk::PipelineLayoutCreateFlags,
    pub set_layouts: Vec<DescriptorSetLayoutCreateInfo>,
    pub push_constant_ranges: Vec<PushConstantRange>,
}
 
impl PipelineCache {
    fn create_descriptor_set_layout_inner<'a>(device: &'a Arc<Device>, map: &'a mut HashMap<DescriptorSetLayoutCreateInfo, Arc<DescriptorSetLayout>>, info: DescriptorSetLayoutCreateInfo) -> &'a Arc<DescriptorSetLayout>{
        map.entry(info).or_insert_with_key(|info| {
            let bindings: Vec<_> = info.bindings.iter().map(|binding| vk::DescriptorSetLayoutBinding {
                binding: binding.binding,
                descriptor_type: binding.descriptor_type,
                descriptor_count: binding.descriptor_count,
                stage_flags: binding.stage_flags,
                p_immutable_samplers: std::ptr::null(),
            }).collect();
            let layout = unsafe {
                DescriptorSetLayout::new(device.clone(), &vk::DescriptorSetLayoutCreateInfo::builder()
                .flags(info.flags)
                .bindings(bindings.as_slice())
                .build()).unwrap()
            };
            Arc::new(layout)
        })
    }
    pub fn create_descriptor_set_layout(&mut self, info: DescriptorSetLayoutCreateInfo) -> &Arc<DescriptorSetLayout> {
        Self::create_descriptor_set_layout_inner(&self.device, &mut self.descriptor_set_layouts, info)
    }
    pub fn create_pipeline_layout(&mut self, info: PipelineLayoutCreateInfo) -> &Arc<PipelineLayout> {
        self.pipeline_layouts.entry(info).or_insert_with_key(|info| {
            let sets: Vec<vk::DescriptorSetLayout> = info.set_layouts.iter().map(|layout| {
                let set = Self::create_descriptor_set_layout_inner(&self.device, &mut self.descriptor_set_layouts, layout.clone());
                set.raw()
            }).collect();
            let info = vk::PipelineLayoutCreateInfo {
                flags: info.flags,
                set_layout_count: sets.len() as u32,
                p_set_layouts: sets.as_ptr(),
                push_constant_range_count: info.push_constant_ranges.len() as u32,
                p_push_constant_ranges: info.push_constant_ranges.as_ptr() as *const _,
                ..Default::default()
            };
            let layout = unsafe {
                PipelineLayout::new(self.device.clone(), &info).unwrap()
            };
            Arc::new(layout)
        })
    }
}
