use std::alloc::Layout;

use crate::resources::alloc::{Allocator, MemBuffer, MemoryBlock};

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
    group_base_alignment: u32,
    num_miss: u32,
}
impl SbtHandles {
    fn rgen(&self) -> &[u8] {
        &self.data[0..self.handle_layout.size()]
    }
    fn rmiss(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size() * (index + 1);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    fn hitgroup(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size() * (index + self.num_miss as usize + 1);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }

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
    pub fn compile<'a, RGEN, RMISS, RHIT, RhitIter>(
        &'a self,
        rgen_data: RGEN,
        rmiss_data: impl IntoIterator<Item = RMISS>,
        rhit_data: impl IntoIterator<Item = (usize, RHIT), IntoIter = RhitIter>, // Iterator to HitGroup, HitGroup Parameter
        allocator: impl FnOnce(Layout) -> MemBuffer,
    ) -> Sbt
    where
        RGEN: Copy,
        RMISS: Copy,
        RHIT: Copy,
        RhitIter: ExactSizeIterator<Item = (usize, RHIT)>,
    {
        let rhit_data: RhitIter = rhit_data.into_iter();

        let raygen_layout = self
            .handle_layout
            .extend(Layout::new::<RGEN>())
            .unwrap()
            .0
            .align_to(self.group_base_alignment as usize)
            .unwrap();
        let raymiss_layout_one = self.handle_layout.extend(Layout::new::<RMISS>()).unwrap().0;
        let raymiss_layout = raymiss_layout_one
            .repeat(self.num_miss as usize)
            .unwrap()
            .0
            .align_to(self.group_base_alignment as usize)
            .unwrap();
        let hitgroup_layout_one = self.handle_layout.extend(Layout::new::<RHIT>()).unwrap().0;
        let hitgroup_layout = hitgroup_layout_one
            .repeat(rhit_data.len())
            .unwrap()
            .0
            .align_to(self.group_base_alignment as usize)
            .unwrap();
        let sbt_layout = raygen_layout
            .extend(raymiss_layout)
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
                set_sbt_item(raygen_slice, rgen_data, self.rgen(), &self.handle_layout);
            }
            {
                // Copy rmiss records
                let front = raygen_layout.pad_to_align().size();
                let back = front + raymiss_layout.size();
                let raymiss_slice = &mut target_slice[front..back];
                let mut rmiss_data = rmiss_data.into_iter();
                for i in 0..self.num_miss as usize {
                    let front = raymiss_layout_one.pad_to_align().size() * i;
                    let back = raymiss_layout_one.size() + front;
                    let current_rmiss_slice = &mut raymiss_slice[front..back];
                    let current_rmiss_data =
                        rmiss_data.next().expect("Not enough rmiss data provided");
                    set_sbt_item(
                        current_rmiss_slice,
                        current_rmiss_data,
                        self.rmiss(i),
                        &self.handle_layout,
                    );
                }
            }
            {
                // Copy hitgroup records
                let front = raygen_layout
                    .extend(raymiss_layout)
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
                        self.hitgroup(hitgroup_index),
                        &self.handle_layout,
                    );
                }
            }
        });

        let base_device_address = target_membuffer.get_device_address();
        assert!(base_device_address % self.group_base_alignment as u64 == 0);
        Sbt {
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
            hit_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: base_device_address
                    + raygen_layout.pad_to_align().size() as u64
                    + raymiss_layout.pad_to_align().size() as u64,
                stride: hitgroup_layout_one.pad_to_align().size() as u64,
                size: hitgroup_layout.pad_to_align().size() as u64,
            },
            callable_sbt: vk::StridedDeviceAddressRegionKHR {
                device_address: 0,
                stride: 0,
                size: 0,
            },
        }
    }
}

impl RayTracingPipeline {
    pub fn create_sbt_handles(&self) -> VkResult<SbtHandles> {
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
            group_base_alignment: rtx_properties.shader_group_base_alignment,
            num_miss: self.num_miss,
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
