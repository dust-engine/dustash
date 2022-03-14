#![feature(into_future)]
#![feature(type_alias_impl_trait)]
#![feature(negative_impls)]
#![feature(array_methods)]
#![feature(maybe_uninit_uninit_array)]

use ash::{prelude::VkResult, vk};
use std::{ffi::CStr, ops::Deref, sync::Arc};

pub mod command;
pub mod fence;
pub mod frames;
pub mod queue;
pub mod resources;
pub mod surface;
pub mod swapchain;

pub struct Instance {
    entry: Arc<ash::Entry>,
    instance: ash::Instance,
}

impl Instance {
    pub fn create(entry: Arc<ash::Entry>, info: &vk::InstanceCreateInfo) -> VkResult<Self> {
        // Safety: No Host Syncronization rules for vkCreateInstance.
        let instance = unsafe { entry.create_instance(info, None)? };
        Ok(Instance { entry, instance })
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
        // Safety: Host Syncronization rule for vkDestroyInstance:
        // - Host access to instance must be externally synchronized.
        // - Host access to all VkPhysicalDevice objects enumerated from instance must be externally synchronized.
        // We have &mut self and therefore exclusive control on instance.
        // VkPhysicalDevice created from this Instance may not exist at this point,
        // because PhysicalDevice retains an Arc to Instance.
        // If there still exist a copy of PhysicalDevice, the Instance wouldn't be dropped.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone)]
pub struct PhysicalDevice {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
}

impl PhysicalDevice {
    pub fn enumerate(instance: &Arc<Instance>) -> VkResult<Vec<Self>> {
        // Safety: No Host Syncronization rules for vkEnumeratePhysicalDevices.
        // It should be OK to call this method and obtain multiple copies of VkPhysicalDevice,
        // because nothing except vkDestroyInstance require exclusive access to VkPhysicalDevice.
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let results = physical_devices
            .into_iter()
            .map(|pdevice| PhysicalDevice {
                instance: instance.clone(), // Retain reference to Instance here
                physical_device: pdevice,   // Borrow VkPhysicalDevice from Instance
                                            // Borrow is safe because we retain a reference to Instance here,
                                            // ensuring that Instance wouldn't be dropped as long as the borrow is still there.
            })
            .collect();
        Ok(results)
    }
    pub fn get_queue_family_properties(&self) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.instance
                .get_physical_device_queue_family_properties(self.physical_device)
        }
    }
    pub fn get_physical_device_properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        }
    }
    pub fn get_physical_device_properties2(&self, properties2: &mut vk::PhysicalDeviceProperties2) {
        unsafe {
            self.instance
                .get_physical_device_properties2(self.physical_device, properties2)
        }
    }
    pub fn get_physical_device_features(&self) -> vk::PhysicalDeviceFeatures {
        unsafe {
            self.instance
                .get_physical_device_features(self.physical_device)
        }
    }
    pub fn get_physical_device_features2(&self, features2: &mut vk::PhysicalDeviceFeatures2) {
        unsafe {
            self.instance
                .get_physical_device_features2(self.physical_device, features2)
        }
    }
    pub fn create_device(
        self,
        enabled_layers: &[&CStr],
        enabled_extensions: &[&CStr],
        enabled_features: &vk::PhysicalDeviceFeatures,
    ) -> VkResult<(Arc<Device>, crate::queue::Queues)> {
        let queue_create_info = queue::QueuesCreateInfo::find(&self);
        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_create_info.create_infos.len() as u32,
            p_queue_create_infos: queue_create_info.create_infos.as_ptr(),

            enabled_layer_count: enabled_layers.len() as u32,
            pp_enabled_layer_names: enabled_layers.as_ptr() as *const *const i8,

            enabled_extension_count: enabled_extensions.len() as u32,
            pp_enabled_extension_names: enabled_extensions.as_ptr() as *const *const i8,

            p_enabled_features: enabled_features,
            ..Default::default()
        };

        // Safety: No Host Syncronization rules for VkCreateDevice.
        // Device retains a reference to Instance, ensuring that Instance is dropped later than Device.
        let device = unsafe {
            self.instance
                .create_device(self.physical_device, &create_info, None)?
        };
        let device = Arc::new(Device {
            physical_device: self,
            device,
        });

        let queues = unsafe {
            // Safe because this is only called once per device.
            crate::queue::Queues::from_device(&device, &queue_create_info)
        };
        Ok((device, queues))
    }
}

pub struct Device {
    physical_device: PhysicalDevice,
    device: ash::Device,
}

impl Device {
    pub fn instance(&self) -> &Arc<Instance> {
        &self.physical_device.instance
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
