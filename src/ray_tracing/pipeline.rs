use super::sbt::{HitGroupType, SbtLayout};
use crate::resources::alloc::{Allocator, BufferRequest};
use crate::Device;
use ash::extensions::khr;
use ash::{prelude::VkResult, vk};
use std::io::IntoInnerError;
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
    pub(super) num_miss: u32,
    pub(super) num_hitgroups: u32,
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
                layout.sbt_layout.miss_shaders.len() + layout.sbt_layout.hitgroup_shaders.len()
            })
            .sum::<usize>()
            + sbt_layouts.len();
        let total_num_groups = sbt_layouts
            .iter()
            .map(|layout| layout.sbt_layout.miss_shaders.len() + layout.sbt_layout.hitgroups.len())
            .sum::<usize>()
            + sbt_layouts.len();

        let mut stages: Vec<vk::PipelineShaderStageCreateInfo> =
            Vec::with_capacity(total_num_stages);
        let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> =
            Vec::with_capacity(total_num_groups);

        fn create_stage(
            module: vk::ShaderModule,
            stage: vk::ShaderStageFlags,
            specialization_info: &vk::SpecializationInfo,
        ) -> vk::PipelineShaderStageCreateInfo {
            const SHADER_ENTRY_NAME_MAIN: &'static std::ffi::CStr =
                unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
            vk::PipelineShaderStageCreateInfo::builder()
                .flags(vk::PipelineShaderStageCreateFlags::default())
                .stage(stage)
                .module(module)
                .name(SHADER_ENTRY_NAME_MAIN)
                .specialization_info(specialization_info)
                .build()
        }
        let default_specialization_info = vk::SpecializationInfo::default();

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
                    layout.sbt_layout.raygen_shader,
                    vk::ShaderStageFlags::RAYGEN_KHR,
                    &default_specialization_info,
                ))
                .chain(layout.sbt_layout.miss_shaders.iter().map(|&module| {
                    create_stage(
                        module,
                        vk::ShaderStageFlags::MISS_KHR,
                        &default_specialization_info,
                    )
                }))
                .chain(
                    layout
                        .sbt_layout
                        .hitgroup_shaders
                        .iter()
                        .map(|(stage, module)| {
                            create_stage(*module, *stage, &default_specialization_info)
                        }),
                );
                let stages_range = stages.len()
                    ..stages.len()
                        + 1
                        + layout.sbt_layout.miss_shaders.len()
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
                .chain((1..=layout.sbt_layout.miss_shaders.len()).map(|i| {
                    vk::RayTracingShaderGroupCreateInfoKHR::builder() // Miss Shader
                        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                        .general_shader(i as u32)
                        .intersection_shader(vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                        .build()
                }))
                .chain(layout.sbt_layout.hitgroups.iter().map(|group| {
                    let base = layout.sbt_layout.miss_shaders.len() as u32 + 1;
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
            .map(|(pipeline, layout)| RayTracingPipeline {
                loader: loader.clone(),
                pipeline,
                num_hitgroups: layout.sbt_layout.hitgroups.len() as u32,
                num_miss: layout.sbt_layout.miss_shaders.len() as u32,
            })
            .collect::<Vec<_>>();
        Ok(pipelines)
    }

    /*
    pub fn create_sbt<RGEN, RMISS, RHIT, RHIT_ITER>(
        &self,
        allocator: Arc<Allocator>,
        rgen_data: RGEN,
        rmiss_data: impl IntoIterator<Item = RMISS>,
        rhit_data: impl IntoIterator<Item = RHIT, IntoIter = RHIT_ITER>)
        where RHIT_ITER: ExactSizeIterator<Item = RHIT>
    {
        use gpu_alloc::Request;
        use gpu_alloc_ash::AshMemoryDevice;
        use std::alloc::Layout;
        let rtx_properties = &self.loader.device().physical_device().properties().ray_tracing;

        let total_num_groups = 1 + self.num_hitgroups + self.num_miss;

        // Layout for a single shader handle
        let handle_layout = Layout::from_size_align(
            rtx_properties.shader_group_handle_size as usize,
            rtx_properties.shader_group_handle_alignment as usize,
        )
        .unwrap();

        let raygen_layout = handle_layout
            .extend(Layout::new::<RGEN>())
            .unwrap()
            .0
            .align_to(rtx_properties.shader_group_base_alignment as usize)
            .unwrap();
        let raymiss_layout = handle_layout
            .extend(Layout::new::<RMISS>())
            .unwrap()
            .0
            .repeat(self.num_miss as usize)
            .unwrap()
            .0
            .align_to(rtx_properties.shader_group_base_alignment as usize)
            .unwrap();
        let hitgroup_layout = handle_layout
            .extend(Layout::new::<RHIT>())
            .unwrap()
            .0
            .repeat(rhit_data.into_iter().len())
            .unwrap()
            .0
            .align_to(rtx_properties.shader_group_base_alignment as usize)
            .unwrap();
        let sbt_layout = raygen_layout
            .extend(raymiss_layout)
            .unwrap()
            .0
            .extend(hitgroup_layout)
            .unwrap()
            .0
            .pad_to_align();

        assert_eq!(
            sbt_layout.align(),
            rtx_properties.shader_group_base_alignment as usize
        );

        let sbt_buf = allocator.allocate_buffer(BufferRequest {
            size: sbt_layout.size() as u64,
            alignment: sbt_layout.align() as u64,
            usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS  | vk::BufferUsageFlags::TRANSFER_SRC,
            memory_usage: gpu_alloc::UsageFlags::UPLOAD,
            ..Default::default()
        });
        let sbt_buf = allocator.allocate_buffer(BufferRequest {
            size: sbt_layout.size() as u64,
            alignment: sbt_layout.align() as u64,
            usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS  | vk::BufferUsageFlags::TRANSFER_SRC,
            memory_usage: gpu_alloc::UsageFlags::UPLOAD,
            ..Default::default()
        });
        let device_address =
        assert_eq!(device_address % sbt_layout.align() as u64, 0);
        unsafe {
            let sbt_handles_host_vec = self.loader
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    total_num_groups,
                    rtx_properties.shader_group_handle_size as usize
                        * total_num_groups as usize,
                    // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
                    // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
                )
                .unwrap();
            let mut sbt_handles_host = sbt_handles_host_vec.as_ptr();
            let mut sbt_handles_device = sbt_mem
                .map(AshMemoryDevice::wrap(&*device), 0, sbt_layout.size())
                .unwrap()
                .as_ptr();

            // Copy Raygen
            std::ptr::copy_nonoverlapping(
                sbt_handles_host,
                sbt_handles_device,
                handle_layout.size(),
            );
            sbt_handles_host = sbt_handles_host.add(handle_layout.size());
            sbt_handles_device = sbt_handles_device.add(raygen_layout.pad_to_align().size());

            {
                // Copy RayMiss
                let mut current_device = sbt_handles_device;
                for _ in 0..self.miss_shaders.len() {
                    std::ptr::copy_nonoverlapping(
                        sbt_handles_host,
                        current_device,
                        handle_layout.size(),
                    );
                    sbt_handles_host = sbt_handles_host.add(handle_layout.size());
                    current_device = current_device.add(handle_layout.pad_to_align().size());
                }
                sbt_handles_device = sbt_handles_device.add(raymiss_layout.pad_to_align().size());
            }
            {
                // Copy Hitgroups
                let mut current_device = sbt_handles_device;
                for _ in 0..self.hitgroups.len() {
                    std::ptr::copy_nonoverlapping(
                        sbt_handles_host,
                        sbt_handles_device,
                        handle_layout.size(),
                    );
                    sbt_handles_host = sbt_handles_host.add(handle_layout.size());
                    current_device = current_device.add(handle_layout.pad_to_align().size());
                }
            }

            sbt_mem.unmap(AshMemoryDevice::wrap(&*device));
        }
        Sbt {
            memory: sbt_mem,
            buffer: sbt_buf,
            raygen_shader_binding_tables: vk::StridedDeviceAddressRegionKHR {
                device_address: device_address,
                size: raygen_layout.size() as u64,
                stride: handle_layout.pad_to_align().size() as u64,
            },
            miss_shader_binding_tables: vk::StridedDeviceAddressRegionKHR {
                device_address: device_address + raygen_layout.pad_to_align().size() as u64,
                size: raymiss_layout.size() as u64,
                stride: handle_layout.pad_to_align().size() as u64,
            },
            hit_shader_binding_tables: vk::StridedDeviceAddressRegionKHR {
                device_address: device_address
                    + raygen_layout.pad_to_align().size() as u64
                    + raymiss_layout.pad_to_align().size() as u64,
                size: hitgroup_layout.size() as u64,
                stride: handle_layout.pad_to_align().size() as u64,
            },
            callable_shader_binding_tables: vk::StridedDeviceAddressRegionKHR::default(),
        }
    }
    */
}
