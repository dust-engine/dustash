use std::{collections::BTreeMap, sync::Arc};

use ash::vk;
use rspirv_reflect::{DescriptorInfo, DescriptorType, Reflection};

use crate::{
    descriptor::DescriptorSetLayout,
    ray_tracing::cache::{DescriptorSetLayoutCreateInfo, PipelineCache, PipelineLayoutCreateInfo},
    Device, HasDevice,
};

pub struct Shader {
    device: Arc<Device>,
    pub(crate) module: vk::ShaderModule,
    pub(crate) descriptor_sets: BTreeMap<u32, BTreeMap<u32, DescriptorInfo>>,
}

impl HasDevice for Shader {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl crate::debug::DebugObject for Shader {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::SHADER_MODULE;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.module) }
    }
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Shader").field(&self.module).finish()
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.module, None) }
    }
}

impl Shader {
    pub fn from_spirv(device: Arc<Device>, spirv: &[u8]) -> Self {
        let module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: spirv.len(),
                        p_code: spirv.as_ptr() as *const u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        let reflection = Reflection::new_from_spirv(spirv).unwrap();
        println!("{:?}", reflection.0.entry_points);
        Self {
            device,
            module,
            descriptor_sets: reflection.get_descriptor_sets().unwrap(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpecializedShader {
    pub shader: Arc<Shader>,
    pub specialization: SpecializationInfo,
    pub entry_point: String,
}
impl PartialEq for SpecializedShader {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shader, &other.shader)
            && self.specialization == other.specialization
            && self.entry_point == other.entry_point
    }
}

#[derive(Clone, Default, Debug)]
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
