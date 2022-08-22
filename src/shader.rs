use std::sync::Arc;

use ash::vk;

use crate::{Device, HasDevice};

pub struct Shader {
    device: Arc<Device>,
    pub(crate) module: vk::ShaderModule,
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
    pub fn from_glsl(device: Arc<Device>, glsl: &str) -> Self {
        let shader_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: glsl.as_bytes().len(),
                        p_code: glsl.as_ptr() as *const u32,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        Self {
            device,
            module: shader_module,
        }
    }

    pub fn from_spirv(device: Arc<Device>, spirv: &[u32]) -> Self {
        let shader_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: spirv.len() * std::mem::size_of::<u32>(),
                        p_code: spirv.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        Self {
            device,
            module: shader_module,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpecializedShader {
    pub shader: Arc<Shader>,
    pub specialization: SpecializationInfo,
}
impl PartialEq for SpecializedShader {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shader, &other.shader) && self.specialization == other.specialization
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
