use super::sbt::{HitGroupType, SbtHandles, SbtLayout};
use crate::shader::SpecializedShader;
use crate::{Device, HasDevice};
use ash::extensions::khr;
use ash::{prelude::VkResult, vk};
use std::{ops::Deref, sync::Arc};

pub struct PipelineLayout {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
}

impl HasDevice for PipelineLayout {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl crate::debug::DebugObject for PipelineLayout {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::PIPELINE_LAYOUT;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.layout) }
    }
}

impl PipelineLayout {
    pub fn new(device: Arc<Device>, info: &vk::PipelineLayoutCreateInfo) -> VkResult<Self> {
        unsafe {
            let layout = device.create_pipeline_layout(info, None)?;
            Ok(Self { device, layout })
        }
    }
}
impl PipelineLayout {
    pub fn raw(&self) -> vk::PipelineLayout {
        self.layout
    }
}
impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct RayTracingLoader {
    device: Arc<Device>,
    loader: khr::RayTracingPipeline,
}
impl Deref for RayTracingLoader {
    type Target = khr::RayTracingPipeline;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}
impl crate::HasDevice for RayTracingLoader {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl RayTracingLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::RayTracingPipeline::new(device.instance(), &device);
        Self { device, loader }
    }
}

pub struct RayTracingPipeline {
    pub(super) loader: Arc<RayTracingLoader>,
    pub(super) pipeline: vk::Pipeline,
    pub(super) handles: SbtHandles,
}
impl RayTracingPipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
}
impl HasDevice for RayTracingPipeline {
    fn device(&self) -> &Arc<Device> {
        &self.loader.device
    }
}
impl crate::debug::DebugObject for RayTracingPipeline {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::PIPELINE;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.pipeline) }
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.loader.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

pub struct RayTracingPipelineLayout<'a> {
    pub pipeline_layout: &'a PipelineLayout,
    pub sbt_layout: &'a SbtLayout,
    pub max_recursion_depth: u32,
}
impl RayTracingPipeline {
    pub fn create_many(
        loader: Arc<RayTracingLoader>,
        sbt_layouts: &[RayTracingPipelineLayout],
    ) -> VkResult<Vec<Self>> {
        let total_num_stages = sbt_layouts
            .iter()
            .map(|layout| {
                layout.sbt_layout.miss_shaders.len()
                    + layout.sbt_layout.hitgroup_shaders.len()
                    + layout.sbt_layout.callable_shaders.len()
            })
            .sum::<usize>()
            + sbt_layouts.len();
        let total_num_groups = sbt_layouts
            .iter()
            .map(|layout| {
                layout.sbt_layout.miss_shaders.len()
                    + layout.sbt_layout.hitgroups.len()
                    + layout.sbt_layout.callable_shaders.len()
            })
            .sum::<usize>()
            + sbt_layouts.len();

        // Specifying the capacity for these arrays is important. We must ensure that these vecs are
        // not gonna be reallocated to ensure that the pointer in PipelineShaderStageCreateInfo remains valid.
        let mut stages: Vec<vk::PipelineShaderStageCreateInfo> =
            Vec::with_capacity(total_num_stages);
        let mut specialization_infos: Vec<vk::SpecializationInfo> =
            Vec::with_capacity(total_num_stages);
        let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> =
            Vec::with_capacity(total_num_groups);

        fn create_stage(
            module: &SpecializedShader,
            stage: vk::ShaderStageFlags,
        ) -> (vk::PipelineShaderStageCreateInfo, vk::SpecializationInfo) {
            static SHADER_ENTRY_NAME_MAIN: &std::ffi::CStr =
                unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
            let specialization_info = vk::SpecializationInfo {
                map_entry_count: module.specialization.entries.len() as u32,
                p_map_entries: module.specialization.entries.as_ptr(),
                data_size: module.specialization.data.len(),
                p_data: module.specialization.data.as_ptr() as *const std::ffi::c_void,
            };
            (
                vk::PipelineShaderStageCreateInfo::builder()
                    .flags(vk::PipelineShaderStageCreateFlags::default())
                    .stage(stage)
                    .module(module.shader.module)
                    .name(SHADER_ENTRY_NAME_MAIN)
                    .specialization_info(&specialization_info)
                    .build(),
                specialization_info,
            )
        }
        let create_infos = sbt_layouts
            .iter()
            .map(|layout| {
                assert_eq!(
                    layout.pipeline_layout.device.handle(),
                    loader.device.handle()
                );

                // A list of vk::PipelineShaderStageCreateInfo containing non-repeating shader module stages
                // with one RayGen shader first, multiple RayMiss shaders, and multiple hitgroup shaders.
                let sbt_stages = std::iter::once(create_stage(
                    &layout.sbt_layout.raygen_shader,
                    vk::ShaderStageFlags::RAYGEN_KHR,
                ))
                .chain(
                    layout
                        .sbt_layout
                        .miss_shaders
                        .iter()
                        .map(|module| create_stage(module, vk::ShaderStageFlags::MISS_KHR)),
                )
                .chain(
                    layout
                        .sbt_layout
                        .callable_shaders
                        .iter()
                        .map(|module| create_stage(module, vk::ShaderStageFlags::CALLABLE_KHR)),
                )
                .chain(
                    layout
                        .sbt_layout
                        .hitgroup_shaders
                        .iter()
                        .map(|(stage, module)| create_stage(module, *stage)),
                )
                .map(|(mut create_info, specialization_info)| {
                    // specialization_infos cannot be reallocated, since RayTracingPipelineCreateInfoKHR retains a pointer into this array
                    debug_assert!(specialization_infos.len() < specialization_infos.capacity());
                    specialization_infos.push(specialization_info);
                    create_info.p_specialization_info = specialization_infos.last().unwrap();
                    create_info
                });
                let stages_range = stages.len()
                    ..stages.len()
                        + 1
                        + layout.sbt_layout.miss_shaders.len()
                        + layout.sbt_layout.callable_shaders.len()
                        + layout.sbt_layout.hitgroup_shaders.len();
                debug_assert!(stages.len() < stages.capacity());
                // stages cannot be reallocated, since RayTracingPipelineCreateInfoKHR retains a pointer into this array
                stages.extend(sbt_stages);
                let sbt_groups = std::iter::once(
                    vk::RayTracingShaderGroupCreateInfoKHR::builder() // Raygen Shader
                        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                        .general_shader(0) // Raygen shader is always at index 0
                        .intersection_shader(vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                        .build(),
                )
                .chain((0..layout.sbt_layout.miss_shaders.len()).map(|i| {
                    vk::RayTracingShaderGroupCreateInfoKHR::builder() // Miss Shader
                        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                        .general_shader(i as u32 + 1)
                        .intersection_shader(vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                        .build()
                }))
                .chain((0..layout.sbt_layout.callable_shaders.len()).map(|i| {
                    vk::RayTracingShaderGroupCreateInfoKHR::builder() // Callable Shader
                        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                        .general_shader(i as u32 + 1 + layout.sbt_layout.miss_shaders.len() as u32)
                        .intersection_shader(vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                        .build()
                }))
                .chain(layout.sbt_layout.hitgroups.iter().map(|group| {
                    let base = layout.sbt_layout.miss_shaders.len() as u32
                        + layout.sbt_layout.callable_shaders.len() as u32
                        + 1;
                    let ty = match group.ty {
                        HitGroupType::Procedural => {
                            vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
                        }
                        HitGroupType::Triangles => {
                            vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
                        }
                    };
                    vk::RayTracingShaderGroupCreateInfoKHR::builder() // Miss Shader
                        .ty(ty)
                        .general_shader(vk::SHADER_UNUSED_KHR)
                        .intersection_shader(
                            group
                                .intersection_shader
                                .map_or(vk::SHADER_UNUSED_KHR, |i| i + base),
                        )
                        .any_hit_shader(
                            group
                                .anyhit_shader
                                .map_or(vk::SHADER_UNUSED_KHR, |i| i + base),
                        )
                        .closest_hit_shader(
                            group
                                .closest_hit_shader
                                .map_or(vk::SHADER_UNUSED_KHR, |i| i + base),
                        )
                        .build()
                }));
                let groups_range = groups.len()
                    ..groups.len()
                        + 1 // raygen
                        + layout.sbt_layout.miss_shaders.len()
                        + layout.sbt_layout.callable_shaders.len()
                        + layout.sbt_layout.hitgroups.len();
                // groups cannot be reallocated, since RayTracingPipelineCreateInfoKHR retains a pointer into this array
                debug_assert!(groups.len() < groups.capacity());
                groups.extend(sbt_groups);

                let info = vk::RayTracingPipelineCreateInfoKHR::builder()
                    .flags(vk::PipelineCreateFlags::default())
                    .stages(&stages[stages_range])
                    .groups(&groups[groups_range])
                    .max_pipeline_ray_recursion_depth(layout.max_recursion_depth)
                    .layout(layout.pipeline_layout.layout)
                    .build();
                info
            })
            .collect::<Vec<_>>();

        let results = unsafe {
            loader.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &create_infos,
                None,
            )?
        };
        let pipelines = results
            .into_iter()
            .zip(sbt_layouts.iter())
            .map(|(pipeline, layout)| {
                Ok(RayTracingPipeline {
                    loader: loader.clone(),
                    pipeline,
                    handles: SbtHandles::new(
                        &loader,
                        pipeline,
                        layout.sbt_layout.miss_shaders.len() as u32,
                        layout.sbt_layout.callable_shaders.len() as u32,
                        layout.sbt_layout.hitgroups.len() as u32,
                    )?,
                })
            })
            .try_collect::<Vec<_>>()?;
        Ok(pipelines)
    }
}
