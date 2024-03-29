mod cache;
pub mod layout_cache;
use crate::{
    command::recorder::CommandRecorder, descriptor::DescriptorSetLayout, Device, HasDevice,
};
use ash::{prelude::VkResult, vk};
pub use cache::*;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Binding {
    pub ty: vk::DescriptorType,
    pub count: u32,
    pub shader_read_stage_flags: vk::ShaderStageFlags,
    pub shader_write_stage_flags: vk::ShaderStageFlags,
}

pub struct PipelineLayout {
    device: Arc<Device>,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) descriptor_sets: Vec<(BTreeMap<u32, Binding>, Arc<DescriptorSetLayout>)>,
}

pub trait Pipeline: Send + Sync {
    fn bind_point(&self) -> vk::PipelineBindPoint;
    fn layout(&self) -> &PipelineLayout;
    fn binding(&self, descriptor_id: u32, binding_id: u32) -> Option<&Binding> {
        self.layout()
            .descriptor_sets
            .get(descriptor_id as usize)
            .and_then(|(a, _)| a.get(&binding_id))
    }
    fn raw(&self) -> vk::Pipeline;
    fn arc_type_erased(self: Arc<Self>) -> Arc<dyn Send + Sync>;
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
    /// Although PipelineLayoutCreateInfo contains references to descriptor sets, we don't actually need to keep those descriptor sets alive.
    /// This is because the descriptor sets are internally reference counted by the driver if the PipelineLayout actually need to keep them alive.
    pub unsafe fn new(
        device: Arc<Device>,
        info: &vk::PipelineLayoutCreateInfo,
        descriptor_sets: Vec<(BTreeMap<u32, Binding>, Arc<DescriptorSetLayout>)>,
    ) -> VkResult<Self> {
        let layout = device.create_pipeline_layout(info, None)?;
        Ok(Self {
            device,
            layout,
            descriptor_sets,
        })
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

pub mod utils {
    use super::*;
    use crate::shader::Shader;

    /// Utility class for collecting descriptor set from multiple shaders
    pub struct ShaderDescriptorSetCollection(BTreeMap<u32, BTreeMap<u32, Binding>>);

    impl ShaderDescriptorSetCollection {
        pub fn new() -> Self {
            Self(BTreeMap::new())
        }
        pub fn merge(&mut self, other: &Shader, stage: vk::ShaderStageFlags) {
            fn shader_descriptor_info_to_ty_count(
                info: &rspirv_reflect::DescriptorInfo,
            ) -> (vk::DescriptorType, u32) {
                use rspirv_reflect::BindingCount::*;
                let other_ty = vk::DescriptorType::from_raw(info.ty.0 as i32);
                let other_binding_count = match info.binding_count {
                    One => 1,
                    StaticSized(count) => count as u32,
                    Unbounded => u32::MAX,
                };
                (other_ty, other_binding_count)
            }
            let mut conflicted: bool = false;
            for (other_descriptor_set_id, other_descriptor_set) in other.descriptor_sets.iter() {
                self.0
                    .entry(*other_descriptor_set_id)
                    .and_modify(|this_descriptor_set| {
                        for (other_binding_id, other_binding) in other_descriptor_set {
                            let (other_ty, other_binding_count) =
                                shader_descriptor_info_to_ty_count(other_binding);

                            this_descriptor_set
                                .entry(*other_binding_id)
                                .and_modify(|this| {
                                    if this.ty != other_ty || this.count != other_binding_count {
                                        conflicted = true;
                                    }
                                    this.shader_read_stage_flags |= stage;
                                    this.shader_write_stage_flags |= stage;
                                })
                                .or_insert_with(|| Binding {
                                    ty: other_ty,
                                    count: other_binding_count,
                                    shader_read_stage_flags: vk::ShaderStageFlags::empty(),
                                    shader_write_stage_flags: vk::ShaderStageFlags::empty(),
                                });
                        }
                    })
                    .or_insert_with(|| {
                        other_descriptor_set
                            .iter()
                            .map(|(index, info)| {
                                let (other_ty, other_binding_count) =
                                    shader_descriptor_info_to_ty_count(info);
                                let binding = Binding {
                                    ty: other_ty,
                                    count: other_binding_count,
                                    shader_read_stage_flags: stage,
                                    shader_write_stage_flags: stage,
                                };
                                (*index, binding)
                            })
                            .collect()
                    });
            }
            assert!(!conflicted); // TODO: Better error handling
        }
        // Chech that the descriptor ids are consequtive
        pub fn flatten(self) -> impl ExactSizeIterator<Item = BTreeMap<u32, Binding>> {
            let mut current_index: u32 = 0;
            for (id, _) in self.0.iter() {
                current_index += 1;
                assert_eq!(
                    *id, current_index,
                    "Descriptor set indexes must be consequtive"
                );
            }
            self.0
                .into_iter()
                .map(|(_descriptor_set_index, descriptor_set)| descriptor_set)
        }
    }
}
