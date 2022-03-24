use std::alloc::Layout;

use crate::resources::alloc::MemBuffer;

use super::pipeline::RayTracingPipeline;
use ash::{prelude::VkResult, vk};

pub struct SbtLayout {
    pub(super) raygen_shader: vk::ShaderModule,
    pub(super) miss_shaders: Vec<vk::ShaderModule>,

    /// A list of non-repeating vk::ShaderModule, alongside their flags
    pub(super) hitgroup_shaders: Vec<(vk::ShaderStageFlags, vk::ShaderModule)>,

    /// A list of HitGroupEntry that indexes into hitgroup_shaders
    pub(super) hitgroups: Vec<HitGroupEntry>,
}

#[derive(Clone, Copy)]
pub enum HitGroupType {
    Triangles,
    Procedural,
}
pub struct HitGroup {
    pub ty: HitGroupType,
    pub intersection_shader: Option<vk::ShaderModule>,
    pub anyhit_shader: Option<vk::ShaderModule>,
    pub closest_hit_shader: Option<vk::ShaderModule>,
}
pub(super) struct HitGroupEntry {
    pub(super) ty: HitGroupType,
    pub(super) intersection_shader: Option<u32>,
    pub(super) anyhit_shader: Option<u32>,
    pub(super) closest_hit_shader: Option<u32>,
}

impl SbtLayout {
    pub fn new<'a>(
        raygen_shader: vk::ShaderModule,
        miss_shaders: Vec<vk::ShaderModule>,
        hitgroups: impl ExactSizeIterator<Item = &'a HitGroup>,
    ) -> Self {
        let mut hitgroup_shaders: Vec<(vk::ShaderStageFlags, vk::ShaderModule)> =
            Vec::with_capacity(hitgroups.len() * 3);
        let hitgroup_entries: Vec<HitGroupEntry> = hitgroups
            .map(|hitgroup| {
                let mut entry = HitGroupEntry {
                    ty: hitgroup.ty,
                    intersection_shader: None,
                    anyhit_shader: None,
                    closest_hit_shader: None,
                };
                macro_rules! push_shader {
                    ($shader_type: ident, $shader_stage_flags: expr) => {
                        if let Some(hitgroup_shader) = hitgroup.$shader_type {
                            let index = hitgroup_shaders
                                .iter()
                                .position(|(_, shader)| *shader == hitgroup_shader)
                                .unwrap_or_else(|| {
                                    let index = hitgroup_shaders.len();
                                    hitgroup_shaders.push(($shader_stage_flags, hitgroup_shader));
                                    index
                                });
                            entry.$shader_type = Some(index as u32);
                        }
                    };
                }
                push_shader!(intersection_shader, vk::ShaderStageFlags::INTERSECTION_KHR);
                push_shader!(anyhit_shader, vk::ShaderStageFlags::ANY_HIT_KHR);
                push_shader!(closest_hit_shader, vk::ShaderStageFlags::CLOSEST_HIT_KHR);
                entry
            })
            .collect();
        Self {
            raygen_shader,
            miss_shaders,
            hitgroup_shaders,
            hitgroups: hitgroup_entries,
        }
    }
}

pub struct SbtHandles {
    data: Box<[u8]>,
    handle_layout: Layout,
    group_base_alignment: usize,
}
impl SbtHandles {
    pub fn raygen(&self) -> &[u8] {
        todo!()
    }
}

impl RayTracingPipeline {
    fn create_sbt_handles(&self) -> VkResult<SbtHandles> {
        let total_num_groups = self.num_hitgroups + self.num_miss + 1;
        let rtx_properties = &self
            .loader
            .device()
            .physical_device()
            .properties()
            .ray_tracing;
        let sbt_handles_host_vec = unsafe {
            self.loader
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    total_num_groups,
                    // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
                    // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
                    rtx_properties.shader_group_handle_size as usize * total_num_groups as usize,
                )?
                .into_boxed_slice()
        };
        Ok(SbtHandles {
            data: sbt_handles_host_vec,
            handle_layout: Layout::from_size_align(
                rtx_properties.shader_group_handle_size as usize,
                rtx_properties.shader_group_handle_alignment as usize,
            )
            .unwrap(),
            group_base_alignment: rtx_properties.shader_group_base_alignment as usize,
        })
    }
}


pub struct Sbt {
    buf: MemBuffer,
    raygen_sbt: vk::StridedDeviceAddressRegionKHR,
    miss_sbt: vk::StridedDeviceAddressRegionKHR,
    hit_sbt: vk::StridedDeviceAddressRegionKHR,
    callable_sbt: vk::StridedDeviceAddressRegionKHR,
}

impl Sbt {

}
