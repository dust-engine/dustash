use std::{collections::HashMap, mem::MaybeUninit, sync::Arc};

use ash::vk;

use crate::{Device, PhysicalDevice};

use super::{dispatcher::{submission::FencedSubmission, QueueDispatcher}, Queue};

#[derive(Clone, Copy)]
pub enum QueueType {
    Graphics = 0,
    Compute = 1,
    Transfer = 2,
    SparseBinding = 3,
}

impl QueueType {
    pub fn priority(&self) -> &'static f32{
        [&QUEUE_PRIORITY_HIGH,&QUEUE_PRIORITY_HIGH, &QUEUE_PRIORITY_MID, &QUEUE_PRIORITY_LOW][*self as usize]
    }
}

pub struct Queues {
    queues: Vec<QueueDispatcher>,
    queue_type_to_dispatcher: [usize; 4],
}

impl Queues {
    pub fn of_type(&self, ty: QueueType) -> &QueueDispatcher {
        let i = self.queue_type_to_dispatcher[ty as usize];
        &self.queues[i]
    }
}

impl Queues {
    // Safety: Can only be called once for each device.
    pub(crate) unsafe fn from_device(device: &Arc<Device>, create_info: &QueuesCreateInfo) -> Self {
        let queue_dispatchers: Vec<QueueDispatcher> = create_info.queue_create_infos().iter().map(|queue_create_info| {
            let queue = device.get_device_queue(queue_create_info.queue_family_index, 0); // We always just create at most one queue for each queue family
            let queue = Queue {
                device: device.clone(),
                queue,
                family_index: queue_create_info.queue_family_index,
            };
            QueueDispatcher::new(queue)
        })
        .collect();
        Queues {
            queues: queue_dispatchers,
            queue_type_to_dispatcher: create_info.queue_type_to_queues_index
        }
    }
}

pub struct QueuesCreateInfo {
    queue_type_to_family: [u32; 4], // Four queue types

    create_infos: [MaybeUninit<vk::DeviceQueueCreateInfo>; 4], // Max 4 queues created. Queue family and index.
    num_queues: u32,
    queue_type_to_queues_index: [usize; 4], // Given QueueType, gives the index of the queue.
}

const QUEUE_PRIORITY_HIGH: f32 = 1.0;
const QUEUE_PRIORITY_MID: f32 = 0.5;
const QUEUE_PRIORITY_LOW: f32 = 1.0;

impl QueuesCreateInfo {
    pub fn find(physical_device: &PhysicalDevice) -> QueuesCreateInfo {
        let available_queue_family = physical_device.get_queue_family_properties();
        Self::find_with_queue_family_properties(&available_queue_family)
    }
    fn find_with_queue_family_properties(available_queue_family: &[vk::QueueFamilyProperties]) -> QueuesCreateInfo {
        let graphics_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_i, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .max_by_key(|&(_i, family)| {
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 1;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        let compute_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_id, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 100;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        let transfer_queue_family = available_queue_family
            .iter()
            .enumerate()
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    priority += 100;
                }
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 10;
                }
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 20;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        let sparse_binding_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_id, family)| family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING))
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    priority -= 1;
                }
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 10;
                }
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 20;
                }
                priority
            })
            .unwrap()
            .0 as u32;

        let queue_type_to_family = [
            graphics_queue_family,
            compute_queue_family,
            transfer_queue_family,
            sparse_binding_queue_family,
        ];
    
        let mut queue_type_to_queues_index: [usize; 4] = [0; 4];
        let mut create_infos: [MaybeUninit<vk::DeviceQueueCreateInfo>; 4] = MaybeUninit::uninit_array(); // (queue_family, QueueType)
        let mut num_queue_families: usize = 0;

        let mut push_queue_type = |ty: QueueType| {
            let create_infos_slice: &mut [vk::DeviceQueueCreateInfo] = unsafe { std::mem::transmute(&mut create_infos[..num_queue_families]) };

            let queue_family = queue_type_to_family[ty as usize];
            let entry = create_infos_slice
                .iter_mut()
                .enumerate()
                .find(|(_, info)| info.queue_family_index == queue_family);
            if let Some((i, entry)) = entry {
                // For this queue type, we've already created a queue.
                // Increase its priority if needed, and record the queue index used for this queue type.
                // This dereference is safe, because entry.p_queue_priorities is assumed to be 'static
                if unsafe{*entry.p_queue_priorities} < *ty.priority() {
                    // Safe because ty.priority() returns a &'static f32
                    entry.p_queue_priorities = ty.priority();
                }
                queue_type_to_queues_index[ty as usize] = i;
            } else {
                create_infos[num_queue_families].write(vk::DeviceQueueCreateInfo {
                    queue_family_index: queue_family,
                    queue_count: 1,
                    // Safe because ty.priority() returns a &'static f32
                    p_queue_priorities: ty.priority(),
                    ..Default::default()
                });
                queue_type_to_queues_index[ty as usize] = num_queue_families;
                num_queue_families += 1;
            }
        };

        push_queue_type(QueueType::Graphics);
        push_queue_type(QueueType::Compute);
        push_queue_type(QueueType::Transfer);
        push_queue_type(QueueType::SparseBinding);

        QueuesCreateInfo {
            queue_type_to_family,
            create_infos,
            num_queues: num_queue_families as u32,
            queue_type_to_queues_index,
        }
    }

    pub fn queue_create_infos(&self) -> &[vk::DeviceQueueCreateInfo] {
        let slice: &[MaybeUninit<vk::DeviceQueueCreateInfo>] = &self.create_infos[0..self.num_queues as usize];

        // Safe. self.num_queues number of queues are initialized.
        unsafe {
            std::mem::transmute(slice)
        }
    }
}
