use std::{alloc::Layout, sync::Arc};

use crate::{resources::alloc::MemBuffer, shader::Shader};

use super::pipeline::{RayTracingLoader, RayTracingPipeline};
use ash::{prelude::VkResult, vk};

pub struct SbtLayout {
    pub(super) raygen_shader: SpecializedShader,
    pub(super) miss_shaders: Box<[SpecializedShader]>,
    pub(super) callable_shaders: Box<[SpecializedShader]>,

    /// A list of non-repeating vk::ShaderModule, alongside their flags
    pub(super) hitgroup_shaders: Vec<(vk::ShaderStageFlags, SpecializedShader)>,

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
    pub intersection_shader: Option<SpecializedShader>,
    pub anyhit_shader: Option<SpecializedShader>,
    pub closest_hit_shader: Option<SpecializedShader>,
}
pub(super) struct HitGroupEntry {
    pub(super) ty: HitGroupType,
    pub(super) intersection_shader: Option<u32>,
    pub(super) anyhit_shader: Option<u32>,
    pub(super) closest_hit_shader: Option<u32>,
}

#[derive(Clone)]
pub struct SpecializationInfo {
    pub(super) data: Vec<u8>,
    pub(super) entries: Vec<vk::SpecializationMapEntry>,
}
impl PartialEq for SpecializationInfo {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .zip(other.entries.iter())
                .all(|(this, other)| {
                    this.constant_id == other.constant_id
                        && this.offset == other.offset
                        && this.size == other.size
                })
    }
}
impl Eq for SpecializationInfo {}
impl SpecializationInfo {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            entries: Vec::new(),
        }
    }
    pub fn push<T: Copy + 'static>(&mut self, constant_id: u32, item: T) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
            panic!("Use push_bool")
        }
        let size = std::mem::size_of::<T>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(&item as *const T as *const u8, target_ptr, size);
            self.data.set_len(self.data.len() + size);
        }
    }
    pub fn push_bool(&mut self, constant_id: u32, item: bool) {
        let size = std::mem::size_of::<vk::Bool32>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let item: vk::Bool32 = if item { vk::TRUE } else { vk::FALSE };
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(
                &item as *const vk::Bool32 as *const u8,
                target_ptr,
                size,
            );
            self.data.set_len(self.data.len() + size);
        }
    }
}

#[derive(Clone)]
pub struct SpecializedShader {
    pub(super) shader: Arc<Shader>,
    pub(super) specialization: Option<SpecializationInfo>,
}
impl PartialEq for SpecializedShader {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shader, &other.shader) && self.specialization == other.specialization
    }
}

impl SbtLayout {
    pub fn new(
        raygen_shader: SpecializedShader,
        miss_shaders: Box<[SpecializedShader]>,
        callable_shaders: Box<[SpecializedShader]>,
        hitgroups: impl ExactSizeIterator<Item = HitGroup>,
    ) -> Self {
        let mut hitgroup_shaders: Vec<(vk::ShaderStageFlags, SpecializedShader)> =
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
                            // Find index of existing shader
                            let index = hitgroup_shaders
                                .iter()
                                .position(|(_, shader)| shader == &hitgroup_shader)
                                .unwrap_or_else(|| {
                                    // Push the new shader into hitgroup_shaders
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
        let rtx_properties = &loader.device().physical_device().properties().ray_tracing;
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
    pub(super) buf: MemBuffer,
    pub(super) raygen_sbt: vk::StridedDeviceAddressRegionKHR,
    pub(super) miss_sbt: vk::StridedDeviceAddressRegionKHR,
    pub(super) hit_sbt: vk::StridedDeviceAddressRegionKHR,
    pub(super) callable_sbt: vk::StridedDeviceAddressRegionKHR,
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
        allocator: impl FnOnce(Layout) -> MemBuffer,
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
        let mut target_membuffer = allocator(sbt_layout);

        fn set_sbt_item<T: Copy>(
            sbt_slice: &mut [u8],
            item: T,
            handle: &[u8],
            handle_layout: &Layout,
        ) {
            sbt_slice[..handle.len()].copy_from_slice(handle);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &item as *const _ as *const u8,
                    sbt_slice[handle_layout.pad_to_align().size()..].as_mut_ptr(),
                    Layout::new::<T>().size(),
                )
            }
        }
        target_membuffer.map_scoped(0, sbt_layout.size(), |target_slice| {
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
                // Copy rmiss records
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

        let base_device_address = target_membuffer.get_device_address();
        assert!(base_device_address % pipeline.handles.group_base_alignment as u64 == 0);
        Sbt {
            pipeline,
            buf: target_membuffer,
            raygen_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: base_device_address,
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
                    + raygen_layout.pad_to_align().size() as u64
                    + raymiss_layout.pad_to_align().size() as u64,
                stride: callable_layout_one.pad_to_align().size() as u64,
                size: callable_layout_one.pad_to_align().size() as u64,
            },
            hit_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: base_device_address
                    + raygen_layout.pad_to_align().size() as u64
                    + raymiss_layout.pad_to_align().size() as u64
                    + callable_layout.pad_to_align().size() as u64,
                stride: hitgroup_layout_one.pad_to_align().size() as u64,
                size: hitgroup_layout.pad_to_align().size() as u64,
            },
        }
    }
}
