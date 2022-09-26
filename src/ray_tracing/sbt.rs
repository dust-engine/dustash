use std::{alloc::Layout, sync::Arc};

use crate::{
    graph::{RenderGraph, RenderGraphContext, ResourceHandle},
    resources::alloc::{Allocator, BufferRequest, MemBuffer},
    shader::SpecializedShader,
    sync::CommandsFuture,
};

use super::pipeline::{RayTracingLoader, RayTracingPipeline};
use crate::HasDevice;
use ash::{prelude::VkResult, vk};
use vk_mem::AllocationCreateFlags;

#[derive(Debug)]
pub struct SbtLayout {
    pub(super) raygen_shader: SpecializedShader,
    pub(super) miss_shaders: Box<[SpecializedShader]>,
    pub(super) callable_shaders: Box<[SpecializedShader]>,

    /// A list of non-repeating vk::ShaderModule, alongside their flags
    pub(super) hitgroup_shaders: Vec<(vk::ShaderStageFlags, SpecializedShader)>,

    /// A list of HitGroupEntry that indexes into hitgroup_shaders
    pub(super) hitgroups: Vec<HitGroupEntry>,
}

#[derive(Clone, Copy, Debug)]
pub enum HitGroupType {
    Triangles,
    Procedural,
}
pub struct HitGroup {
    pub ty: HitGroupType,
    pub intersection_shader: Option<SpecializedShader>,
    pub anyhit_shader: Option<SpecializedShader>,
    pub closest_hit_shader: Option<SpecializedShader>,
}
#[derive(Debug)]
pub(super) struct HitGroupEntry {
    pub(super) ty: HitGroupType,
    pub(super) intersection_shader: Option<u32>,
    pub(super) anyhit_shader: Option<u32>,
    pub(super) closest_hit_shader: Option<u32>,
}

impl SbtLayout {
    pub fn new(
        raygen_shader: SpecializedShader,
        miss_shaders: Box<[SpecializedShader]>,
        callable_shaders: Box<[SpecializedShader]>,
        hitgroups: &[HitGroup],
    ) -> Self {
        let mut hitgroup_shaders: Vec<(vk::ShaderStageFlags, SpecializedShader)> =
            Vec::with_capacity(hitgroups.len() * 3);
        let hitgroup_entries: Vec<HitGroupEntry> = hitgroups
            .into_iter()
            .map(|hitgroup| {
                let mut entry = HitGroupEntry {
                    ty: hitgroup.ty,
                    intersection_shader: None,
                    anyhit_shader: None,
                    closest_hit_shader: None,
                };
                macro_rules! push_shader {
                    ($shader_type: ident, $shader_stage_flags: expr) => {
                        if let Some(hitgroup_shader) = hitgroup.$shader_type.as_ref() {
                            // Find index of existing shader
                            let index = hitgroup_shaders
                                .iter()
                                .position(|(_, shader)| shader == hitgroup_shader)
                                .unwrap_or_else(|| {
                                    // Push the new shader into hitgroup_shaders
                                    let index = hitgroup_shaders.len();
                                    hitgroup_shaders
                                        .push(($shader_stage_flags, hitgroup_shader.clone()));
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
            callable_shaders,
            hitgroup_shaders,
            hitgroups: hitgroup_entries,
        }
    }
}

pub struct SbtHandles {
    data: Box<[u8]>,
    handle_layout: Layout,
    group_base_alignment: u32,
    num_miss: u32,
    num_callable: u32,
    num_hitgroup: u32,
}
impl SbtHandles {
    pub fn new(
        loader: &RayTracingLoader,
        pipeline: vk::Pipeline,
        num_miss: u32,
        num_callable: u32,
        num_hitgroup: u32,
    ) -> VkResult<SbtHandles> {
        let total_num_groups = num_hitgroup + num_miss + num_callable + 1;
        let rtx_properties = &loader.physical_device().properties().ray_tracing;
        let sbt_handles_host_vec = unsafe {
            loader
                .get_ray_tracing_shader_group_handles(
                    pipeline,
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
            group_base_alignment: rtx_properties.shader_group_base_alignment,
            num_miss,
            num_callable,
            num_hitgroup,
        })
    }

    fn rgen(&self) -> &[u8] {
        &self.data[0..self.handle_layout.size()]
    }
    fn rmiss(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size() * (index + 1);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    fn callable(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size() * (index + 1 + self.num_miss as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    fn hitgroup(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size()
            * (index + self.num_miss as usize + self.num_callable as usize + 1);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
}

pub struct Sbt {
    pub(super) pipeline: Arc<RayTracingPipeline>,
    pub(super) raygen_sbt: vk::StridedDeviceAddressRegionKHR,
    pub(super) miss_sbt: vk::StridedDeviceAddressRegionKHR,
    pub(super) hit_sbt: vk::StridedDeviceAddressRegionKHR,
    pub(super) callable_sbt: vk::StridedDeviceAddressRegionKHR,
    total_size: u64,

    buf_handle: ResourceHandle<MemBuffer>,
    staging_handle: Option<ResourceHandle<MemBuffer>>,
}

impl std::fmt::Debug for Sbt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sbt")
            .field("raygen_sbt", &self.raygen_sbt)
            .field("miss_sbt", &self.miss_sbt)
            .field("hit_sbt", &self.hit_sbt)
            .field("callable_sbt", &self.callable_sbt)
            .finish()
    }
}

impl Sbt {
    /// Compile the SBT data and put them into the allocated buffer.
    ///
    /// # Arguments
    /// * `rgen_data` - Arguments to be passed to the ray generation shader.
    /// * `rmiss_data` - Arguments to be passed to each of the ray miss shader.
    /// * `rhit_data` - A tuple with the first item being the HitGroup index and
    /// the second item being the arguments to be passed to each of the hit group shaders.
    ///
    /// # Examples in shaders
    /// ```glsl
    /// layout(shaderRecordKHR) buffer SBTData {
    ///     uint32_t material_id;
    /// };
    /// ```
    pub fn new<RGEN, RMISS, RHIT, RCALLABLE, RhitIter>(
        pipeline: Arc<RayTracingPipeline>,
        rgen_data: RGEN,
        rmiss_data: impl IntoIterator<Item = RMISS>,
        callable_data: impl IntoIterator<Item = RCALLABLE>,
        rhit_data: impl IntoIterator<Item = (usize, RHIT), IntoIter = RhitIter>, // Iterator to HitGroup, HitGroup Parameter
        allocator: &Arc<Allocator>,
        render_graph: &mut RenderGraph,
    ) -> Sbt
    where
        RGEN: Copy,
        RMISS: Copy,
        RHIT: Copy,
        RCALLABLE: Copy,
        RhitIter: ExactSizeIterator<Item = (usize, RHIT)>,
    {
        let rhit_data: RhitIter = rhit_data.into_iter();

        let raygen_layout = pipeline
            .handles
            .handle_layout
            .extend(Layout::new::<RGEN>())
            .unwrap()
            .0
            .align_to(pipeline.handles.group_base_alignment as usize)
            .unwrap();
        let raymiss_layout_one = pipeline
            .handles
            .handle_layout
            .extend(Layout::new::<RMISS>())
            .unwrap()
            .0;
        let raymiss_layout = raymiss_layout_one
            .repeat(pipeline.handles.num_miss as usize)
            .unwrap()
            .0
            .align_to(pipeline.handles.group_base_alignment as usize)
            .unwrap();
        let callable_layout_one = pipeline
            .handles
            .handle_layout
            .extend(Layout::new::<RCALLABLE>())
            .unwrap()
            .0;
        let callable_layout = callable_layout_one
            .repeat(pipeline.handles.num_callable as usize)
            .unwrap()
            .0
            .align_to(pipeline.handles.group_base_alignment as usize)
            .unwrap();
        let hitgroup_layout_one = pipeline
            .handles
            .handle_layout
            .extend(Layout::new::<RHIT>())
            .unwrap()
            .0;
        let hitgroup_layout = hitgroup_layout_one
            .repeat(rhit_data.len())
            .unwrap()
            .0
            .align_to(pipeline.handles.group_base_alignment as usize)
            .unwrap();
        let sbt_layout = raygen_layout
            .extend(raymiss_layout)
            .unwrap()
            .0
            .extend(callable_layout)
            .unwrap()
            .0
            .extend(hitgroup_layout)
            .unwrap()
            .0
            .pad_to_align();
        let mut target_membuffer = allocator
            .allocate_buffer(&BufferRequest {
                size: sbt_layout.size() as u64,
                alignment: sbt_layout.align() as u64,
                usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                scenario: crate::resources::alloc::MemoryAllocScenario::DynamicStorage,
                allocation_flags: AllocationCreateFlags::empty(),
                ..Default::default()
            })
            .unwrap();
        let mut staging_buffer = if !target_membuffer
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            Some(
                allocator
                    .allocate_buffer(&BufferRequest {
                        size: sbt_layout.size() as u64,
                        alignment: 0,
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        scenario: crate::resources::alloc::MemoryAllocScenario::StagingBuffer,
                        allocation_flags: AllocationCreateFlags::empty(),
                        ..Default::default()
                    })
                    .unwrap(),
            )
        } else {
            None
        };
        let buffer_to_write = staging_buffer.as_mut().unwrap_or(&mut target_membuffer);

        fn set_sbt_item<T: Copy>(
            sbt_slice: &mut [u8],
            inline_data: T,
            handle: &[u8],
            handle_layout: &Layout,
        ) {
            sbt_slice[..handle.len()].copy_from_slice(handle); // Copy handle
            unsafe {
                // Copy inline data
                std::ptr::copy_nonoverlapping(
                    &inline_data as *const _ as *const u8,
                    sbt_slice[handle_layout.pad_to_align().size()..].as_mut_ptr(),
                    Layout::new::<T>().size(),
                )
            }
        }
        buffer_to_write.map_scoped(|target_slice| {
            {
                // Copy raygen records
                let raygen_slice = &mut target_slice[0..raygen_layout.size()];
                set_sbt_item(
                    raygen_slice,
                    rgen_data,
                    pipeline.handles.rgen(),
                    &pipeline.handles.handle_layout,
                );
            }
            {
                // Copy rmiss records
                let front = raygen_layout.pad_to_align().size();
                let back = front + raymiss_layout.size();
                let raymiss_slice = &mut target_slice[front..back];
                let mut rmiss_data = rmiss_data.into_iter();
                for i in 0..pipeline.handles.num_miss as usize {
                    let front = raymiss_layout_one.pad_to_align().size() * i;
                    let back = raymiss_layout_one.size() + front;
                    let current_rmiss_slice = &mut raymiss_slice[front..back];
                    let current_rmiss_data =
                        rmiss_data.next().expect("Not enough rmiss data provided");
                    set_sbt_item(
                        current_rmiss_slice,
                        current_rmiss_data,
                        pipeline.handles.rmiss(i),
                        &pipeline.handles.handle_layout,
                    );
                }
            }
            {
                // Copy callable records
                let front = raygen_layout
                    .extend(raymiss_layout)
                    .unwrap()
                    .0
                    .pad_to_align()
                    .size();
                let back = front + callable_layout.size();
                let callable_slice = &mut target_slice[front..back];
                let mut callable_data = callable_data.into_iter();
                for i in 0..pipeline.handles.num_callable as usize {
                    let front = callable_layout_one.pad_to_align().size() * i;
                    let back = callable_layout_one.size() + front;
                    let current_callable_slice = &mut callable_slice[front..back];
                    let current_callable_data = callable_data
                        .next()
                        .expect("Not enough callable data provided");
                    set_sbt_item(
                        current_callable_slice,
                        current_callable_data,
                        pipeline.handles.callable(i),
                        &pipeline.handles.handle_layout,
                    );
                }
            }
            {
                // Copy hitgroup records
                let front = raygen_layout
                    .extend(raymiss_layout)
                    .unwrap()
                    .0
                    .extend(callable_layout)
                    .unwrap()
                    .0
                    .pad_to_align()
                    .size();
                let back = front + hitgroup_layout.size();
                let hitgroup_slice = &mut target_slice[front..back];
                for (i, (hitgroup_index, current_hitgroup_data)) in rhit_data.enumerate() {
                    let front = hitgroup_layout_one.pad_to_align().size() * i;
                    let back = hitgroup_layout_one.size() + front;
                    let current_hitgroup_slice = &mut hitgroup_slice[front..back];
                    set_sbt_item(
                        current_hitgroup_slice,
                        current_hitgroup_data,
                        pipeline.handles.hitgroup(hitgroup_index),
                        &pipeline.handles.handle_layout,
                    );
                }
            }
        });
        let base_device_address = target_membuffer.device_address();

        let target_buffer_handle = render_graph.import(target_membuffer);
        let staging_buffer_handle = if let Some(staging_buffer) = staging_buffer {
            Some(render_graph.import(staging_buffer))
        } else {
            None
        };

        assert!(base_device_address % pipeline.handles.group_base_alignment as u64 == 0);
        let sbt = Sbt {
            pipeline,
            raygen_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: base_device_address,
                // VUID-vkCmdTraceRaysKHR-size-04023: The size member of pRayGenShaderBindingTable must be equal to its stride member
                // TODO: VUID-vkCmdTraceRaysKHR-pRayG enShaderBindingTable-03680: If the buffer from whichpRayGenShaderBindingTable->deviceAddress
                // was queried is non-sparse then it must be bound completely and contiguously to a single VkDeviceMemory object
                stride: raygen_layout.pad_to_align().size() as u64,
                size: raygen_layout.pad_to_align().size() as u64,
            },
            miss_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: base_device_address + raygen_layout.pad_to_align().size() as u64,
                stride: raymiss_layout_one.pad_to_align().size() as u64,
                size: raymiss_layout.pad_to_align().size() as u64,
            },
            callable_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: base_device_address
                    + raygen_layout
                        .extend(raymiss_layout)
                        .unwrap()
                        .0
                        .pad_to_align()
                        .size() as u64,
                stride: callable_layout_one.pad_to_align().size() as u64,
                size: callable_layout_one.pad_to_align().size() as u64,
            },
            hit_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: base_device_address
                    + raygen_layout
                        .extend(raymiss_layout)
                        .unwrap()
                        .0
                        .extend(callable_layout)
                        .unwrap()
                        .0
                        .pad_to_align()
                        .size() as u64,
                stride: hitgroup_layout_one.pad_to_align().size() as u64,
                size: hitgroup_layout.pad_to_align().size() as u64,
            },
            total_size: sbt_layout.size() as u64,
            buf_handle: target_buffer_handle,
            staging_handle: staging_buffer_handle,
        };
        sbt
    }

    pub fn transfer(&self, ctx: &mut RenderGraphContext) {
        if let Some(staging) = self.staging_handle {
            ctx.copy_buffer(
                staging,
                self.buf_handle,
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: self.total_size,
                },
            )
        }
    }
}

// | raygen: 64 bytes | rmiss: 64 bytes | callable: 2112 bytes | hitgroup: 64 bytes |
