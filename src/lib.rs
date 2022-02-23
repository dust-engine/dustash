use ash::{prelude::VkResult, vk};
use std::{ops::Deref, sync::Arc, collections::HashMap};

pub mod command;
pub mod queue;

use queue::Queue;

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

pub struct PhysicalDevice {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
}

impl PhysicalDevice {
    pub fn enumerate(instance: Arc<Instance>) -> VkResult<Vec<Self>> {
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
    pub fn create_device(&self, info: &vk::DeviceCreateInfo) -> VkResult<(Arc<Device>, HashMap<u32, Vec<Queue>>)> {
        // Safety: No Host Syncronization rules for VkCreateDevice.
        // Device retains a reference to Instance, ensuring that Instance is dropped later than Device.
        let device = unsafe {
            self.instance
                .create_device(self.physical_device, info, None)?
        };
        let device = Arc::new(Device {
            instance: self.instance.clone(),
            device,
        });

        // Safety: We create vkQueues from vk::DeviceCreateInfo, ensure that there's only one copy of Queue for each queue.
        // It is not OK to create multiple copies of Queue, since queue operations require exclusive access to Queue.
        let result: HashMap<u32, Vec<Queue>> = unsafe {
            let queue_create_infos = std::slice::from_raw_parts(info.p_queue_create_infos, info.queue_create_info_count as usize);
            queue_create_infos.iter().map(|create_info| {
                let queue_family_index = create_info.queue_family_index;
                let queues: Vec<Queue> = (0..create_info.queue_count).map(|queue_index| {
                    let queue = device.get_device_queue(queue_family_index, queue_index);
                    Queue {
                        device: device.clone(),
                        queue
                    }
                }).collect::<Vec<Queue>>();
                (queue_family_index, queues)
            }).collect()
        };
        Ok((device, result))
    }
}

pub struct Device {
    instance: Arc<Instance>,
    device: ash::Device,
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
