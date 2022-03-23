#![feature(into_future)]
#![feature(type_alias_impl_trait)]
#![feature(negative_impls)]
#![feature(array_methods)]
#![feature(maybe_uninit_uninit_array)]
#![feature(const_for)]
#![feature(const_option_ext)]
#![feature(alloc_layout_extra)]
#![feature(int_roundings)]
#![feature(core_ffi_c)]

use ash::{prelude::VkResult, vk};
use std::{ops::Deref, sync::Arc};

pub mod command;
mod debug;
pub use debug::DebugUtilsMessenger;
pub mod accel_struct;
pub mod fence;
pub mod frames;
mod physical_device;
pub mod queue;
pub mod resources;
pub mod surface;
pub mod swapchain;
pub use physical_device::*;
pub mod descriptor;
mod ray_tracing;

pub struct Instance {
    entry: Arc<ash::Entry>,
    instance: ash::Instance,
    debug_utils: DebugUtilsMessenger,
}

impl Instance {
    pub fn create(entry: Arc<ash::Entry>, info: &vk::InstanceCreateInfo) -> VkResult<Self> {
        // Safety: No Host Syncronization rules for vkCreateInstance.
        let mut instance = unsafe { entry.create_instance(info, None)? };
        let debug_utils = DebugUtilsMessenger::new(&entry, &mut instance)?;
        Ok(Instance {
            entry,
            instance,
            debug_utils,
        })
    }
    pub fn entry(&self) -> &Arc<ash::Entry> {
        &self.entry
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        tracing::info!(instance = ?self.instance.handle(), "drop instance");
        // Safety: Host Syncronization rule for vkDestroyInstance:
        // - Host access to instance must be externally synchronized.
        // - Host access to all VkPhysicalDevice objects enumerated from instance must be externally synchronized.
        // We have &mut self and therefore exclusive control on instance.
        // VkPhysicalDevice created from this Instance may not exist at this point,
        // because PhysicalDevice retains an Arc to Instance.
        // If there still exist a copy of PhysicalDevice, the Instance wouldn't be dropped.
        unsafe {
            self.debug_utils
                .debug_utils
                .destroy_debug_utils_messenger(self.debug_utils.messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct Device {
    physical_device: PhysicalDevice,
    device: ash::Device,
}

impl Device {
    pub fn instance(&self) -> &Arc<Instance> {
        &self.physical_device.instance()
    }
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        tracing::info!(device = ?self.device.handle(), "drop deice");
        // Safety: Host Syncronization rule for vkDestroyDevice:
        // - Host access to device must be externally synchronized.
        // - Host access to all VkQueue objects created from device must be externally synchronized
        // We have &mut self and therefore exclusive control on device.
        // VkQueue objects may not exist at this point, because Queue retains an Arc to Device.
        // If there still exist a Queue, the Device wouldn't be dropped.
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
