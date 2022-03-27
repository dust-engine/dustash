use std::sync::Arc;

use ash::{vk, Device};

pub struct Shader {
    device: Arc<Device>,
    module: vk::ShaderModule,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None)
        }
    }
}

impl Shader {
    pub fn from_glsl(device: Arc<Device>, glsl: &str) -> Self {
        let shader_module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo {
                code_size: glsl.as_bytes().len(),
                p_code: glsl.as_ptr() as *const u32,
                ..Default::default()
            }, None).unwrap()
        };
        Self {
            device,
            module: shader_module
        }
    }

    pub fn from_spirv(device: Arc<Device>, spirv: &[u32]) -> Self {
        let shader_module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo {
                code_size: spirv.len() * std::mem::size_of::<u32>(),
                p_code: spirv.as_ptr(),
                ..Default::default()
            }, None).unwrap()
        };
        Self {
            device,
            module: shader_module
        }
    }
}
