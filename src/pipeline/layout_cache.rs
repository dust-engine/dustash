use crate::{descriptor::DescriptorSetLayout, Device};
use ash::vk;
use std::{collections::HashMap, hash::Hash, sync::Arc};

use super::Binding;
use super::PipelineLayout;

pub struct PipelineLayoutCache {
    device: Arc<Device>,
    descriptor_set_layouts: HashMap<DescriptorSetLayoutCreateInfo, Arc<DescriptorSetLayout>>,
    pipeline_layouts: HashMap<PipelineLayoutCreateInfo, Arc<PipelineLayout>>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct DescriptorSetLayoutCreateInfo {
    pub flags: vk::DescriptorSetLayoutCreateFlags,
    /// Sorted by binding index
    pub bindings: Vec<(u32, Binding)>,
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

impl PipelineLayoutCache {
    fn create_descriptor_set_layout_inner<'a>(
        device: &'a Arc<Device>,
        map: &'a mut HashMap<DescriptorSetLayoutCreateInfo, Arc<DescriptorSetLayout>>,
        info: DescriptorSetLayoutCreateInfo,
    ) -> &'a Arc<DescriptorSetLayout> {
        map.entry(info).or_insert_with_key(|info| {
            let bindings: Vec<_> = info
                .bindings
                .iter()
                .map(|(binding_index, binding)| vk::DescriptorSetLayoutBinding {
                    binding: *binding_index,
                    descriptor_type: binding.ty,
                    descriptor_count: binding.count,
                    stage_flags: binding.shader_read_stage_flags | binding.shader_read_stage_flags,
                    p_immutable_samplers: std::ptr::null(),
                })
                .collect();
            let layout = unsafe {
                DescriptorSetLayout::new(
                    device.clone(),
                    &vk::DescriptorSetLayoutCreateInfo::builder()
                        .flags(info.flags)
                        .bindings(bindings.as_slice())
                        .build(),
                )
                .unwrap()
            };
            Arc::new(layout)
        })
    }
    pub fn create_descriptor_set_layout(
        &mut self,
        info: DescriptorSetLayoutCreateInfo,
    ) -> &Arc<DescriptorSetLayout> {
        Self::create_descriptor_set_layout_inner(
            &self.device,
            &mut self.descriptor_set_layouts,
            info,
        )
    }
    pub fn create_pipeline_layout(
        &mut self,
        info: PipelineLayoutCreateInfo,
    ) -> &Arc<PipelineLayout> {
        self.pipeline_layouts
            .entry(info)
            .or_insert_with_key(|info| {
                let sets: Vec<_> = info
                    .set_layouts
                    .iter()
                    .map(|layout| {
                        let set = Self::create_descriptor_set_layout_inner(
                            &self.device,
                            &mut self.descriptor_set_layouts,
                            layout.clone(),
                        );
                        set.clone()
                    })
                    .collect();
                let raw_sets: Vec<vk::DescriptorSetLayout> = sets.iter().map(|a| a.raw()).collect();
                let create_info = vk::PipelineLayoutCreateInfo {
                    flags: info.flags,
                    set_layout_count: raw_sets.len() as u32,
                    p_set_layouts: raw_sets.as_ptr(),
                    push_constant_range_count: info.push_constant_ranges.len() as u32,
                    p_push_constant_ranges: info.push_constant_ranges.as_ptr() as *const _,
                    ..Default::default()
                };
                let descriptor_set_indexes = info
                    .set_layouts
                    .iter()
                    .zip(sets.into_iter())
                    .map(|(layout, set)| {
                        let bindings = layout
                            .bindings
                            .iter()
                            .map(|(index, binding)| (*index, binding.clone()))
                            .collect();
                        (bindings, set)
                    })
                    .collect();
                let layout = unsafe {
                    PipelineLayout::new(self.device.clone(), &create_info, descriptor_set_indexes)
                        .unwrap()
                };
                Arc::new(layout)
            })
    }
}
