use super::sbt::{HitGroupType, SbtHandles, SbtLayout};
use crate::Device;
use crate::ray_tracing::sbt::SpecializedShader;
use ash::extensions::khr;
use ash::{prelude::VkResult, vk};
use std::{ops::Deref, sync::Arc};

pub struct PipelineLayout {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
}

impl PipelineLayout {
    pub fn new(device: Arc<Device>, info: &vk::PipelineLayoutCreateInfo) -> VkResult<Self> {
        unsafe {
            let layout = device.create_pipeline_layout(info, None)?;
            Ok(Self { device, layout })
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
impl RayTracingLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::RayTracingPipeline::new(device.instance(), &device);
        Self { device, loader }
    }
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

pub struct RayTracingPipeline {
    pub(super) loader: Arc<RayTracingLoader>,
    pub(super) pipeline: vk::Pipeline,
    pub(super) handles: SbtHandles,
}
pub struct RayTracingPipelineLayout<'a> {
    pipeline_layout: &'a PipelineLayout,
    sbt_layout: &'a SbtLayout,
    max_recursion_depth: u32,
}
impl RayTracingPipeline {
    pub fn new(
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

        let mut stages: Vec<vk::PipelineShaderStageCreateInfo> =
            Vec::with_capacity(total_num_stages);
        let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> =
            Vec::with_capacity(total_num_groups);

        fn create_stage(
            module: &SpecializedShader,
            stage: vk::ShaderStageFlags,
        ) -> vk::PipelineShaderStageCreateInfo {
            static SHADER_ENTRY_NAME_MAIN: &std::ffi::CStr =
                unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
            let specialization_info = if let Some(info) = module.specialization.as_ref() {
                vk::SpecializationInfo {
                    map_entry_count: info.entries.len() as u32,
                    p_map_entries: info.entries.as_ptr(),
                    data_size: info.data.len(),
                    p_data: info.data.as_ptr() as *const std::ffi::c_void,
                }
            } else {
                vk::SpecializationInfo::default()
            };
            vk::PipelineShaderStageCreateInfo::builder()
                .flags(vk::PipelineShaderStageCreateFlags::default())
                .stage(stage)
                .module(module.shader.module)
                .name(SHADER_ENTRY_NAME_MAIN)
                .specialization_info(&specialization_info)
                .build()
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
                .chain(layout.sbt_layout.miss_shaders.iter().map(|module| {
                    create_stage(
                        module,
                        vk::ShaderStageFlags::MISS_KHR,
                    )
                }))
                .chain(layout.sbt_layout.callable_shaders.iter().map(|module| {
                    create_stage(
                        module,
                        vk::ShaderStageFlags::CALLABLE_KHR,
                    )
                }))
                .chain(
                    layout
                        .sbt_layout
                        .hitgroup_shaders
                        .iter()
                        .map(|(stage, module)| {
                            create_stage(module, *stage)
                        }),
                );
                let stages_range = stages.len()
                    ..stages.len()
                        + 1
                        + layout.sbt_layout.miss_shaders.len()
                        + layout.sbt_layout.callable_shaders.len()
                        + layout.sbt_layout.hitgroup_shaders.len();
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
                .chain((0..=layout.sbt_layout.callable_shaders.len()).map(|i| {
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
                let groups_range = stages.len()
                    ..stages.len()
                        + 1
                        + layout.sbt_layout.miss_shaders.len()
                        + layout.sbt_layout.callable_shaders.len()
                        + layout.sbt_layout.hitgroups.len();
                groups.extend(sbt_groups);

                vk::RayTracingPipelineCreateInfoKHR::builder()
                    .flags(vk::PipelineCreateFlags::default())
                    .stages(&stages[stages_range])
                    .groups(&groups[groups_range])
                    .max_pipeline_ray_recursion_depth(layout.max_recursion_depth)
                    .layout(layout.pipeline_layout.layout)
                    .build()
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
