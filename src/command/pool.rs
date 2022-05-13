use ash::{prelude::VkResult, vk};
use std::sync::{Arc, Mutex};

use crate::Device;

pub struct CommandPool {
    device: Arc<Device>,
    pub(crate) pool: Mutex<vk::CommandPool>,
    pub(crate) queue_family_index: u32,
}

impl crate::HasDevice for CommandPool {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl CommandPool {
    pub fn new(
        device: Arc<Device>,
        flags: vk::CommandPoolCreateFlags,
        queue_family_index: u32,
    ) -> VkResult<CommandPool> {
        // Safety: No Host Syncronization rules for vkCreateCommandPool.
        unsafe {
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags,
                    queue_family_index,
                    ..Default::default()
                },
                None,
            )?;
            let pool = Mutex::new(pool);
            Ok(CommandPool {
                device,
                pool,
                queue_family_index,
            })
        }
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
    pub fn allocate_one(self: &Arc<CommandPool>) -> VkResult<CommandBuffer> {
        // Safety: Host Syncronization rule for vkAllocateCommandBuffers:
        // - Host access to pAllocateInfo->commandPool must be externally synchronized.
        // self.pool is protected behind a mutex.
        let pool = self.pool.lock().unwrap();
        let mut buffer = vk::CommandBuffer::null();
        let result = unsafe {
            (self.device.fp_v1_0().allocate_command_buffers)(
                self.device.handle(),
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1)
                    .build(),
                &mut buffer,
            )
        };
        drop(pool);
        result.result_with_success(CommandBuffer {
            pool: self.clone(),
            buffer,
        })
    }
    pub fn allocate<const N: usize>(self: &Arc<CommandPool>) -> VkResult<[CommandBuffer; N]> {
        // Safety: Host Syncronization rule for vkAllocateCommandBuffers:
        // - Host access to pAllocateInfo->commandPool must be externally synchronized.
        // self.pool is protected behind a mutex.
        let mut buffers = [vk::CommandBuffer::null(); N];
        let pool = self.pool.lock().unwrap();
        let result = unsafe {
            (self.device.fp_v1_0().allocate_command_buffers)(
                self.device.handle(),
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(N.try_into().unwrap())
                    .build(),
                buffers.as_mut_ptr(),
            )
        };
        drop(pool);
        result.result_with_success(buffers.map(|buffer| CommandBuffer {
            pool: self.clone(),
            buffer,
        }))
    }
    pub fn allocate_n(self: &Arc<CommandPool>, n: u32) -> VkResult<Vec<CommandBuffer>> {
        // Safety: Host Syncronization rule for vkAllocateCommandBuffers:
        // - Host access to pAllocateInfo->commandPool must be externally synchronized.
        // self.pool is protected behind a mutex.
        let pool = self.pool.lock().unwrap();
        let buffers = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(n)
                    .build(),
            )?
        };
        drop(pool);
        Ok(buffers
            .into_iter()
            .map(|buffer| CommandBuffer {
                pool: self.clone(),
                buffer,
            })
            .collect())
    }

    pub fn reset(&self, release_resources: bool) -> VkResult<()> {
        let flags = if release_resources {
            vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandPoolResetFlags::empty()
        };
        let pool = self.pool.lock().unwrap();

        // Safety: Host Syncronization rule for vkAllocateCommandBuffers:
        // - Host access to commandPool must be externally synchronized.
        // self.pool is protected behind a mutex.
        unsafe { self.device.reset_command_pool(*pool, flags) }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        tracing::info!(command_pool = ?self.pool, "drop command pool");
        // Safety: Host Syncronization rule for vkDestroyCommandPool:
        // - Host access to commandPool must be externally synchronized
        // We have &mut self and thus ownership on commandPool.
        unsafe {
            self.device
                .destroy_command_pool(*self.pool.get_mut().unwrap(), None);
        }
    }
}

// vk::CommandBuffer in Initial state.
// When the CommandPool was reset as a whole, command buffers in Initial states would remain in that state.
pub struct CommandBuffer {
    pub(crate) pool: Arc<CommandPool>,
    pub(crate) buffer: vk::CommandBuffer,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        tracing::debug!(command_buffer = ?self.buffer, "drop command buffer");
        // Safety: Host Syncronization rule for vkFreeCommandBuffers:
        // - Host access to commandPool must be externally synchronized
        // - Host access to each member of pCommandBuffers must be externally synchronized
        // We have &mut self and thus ownership on self.buffer.
        // self.pool.pool was protected behind a mutex.
        let pool = self.pool.pool.lock().unwrap();
        unsafe {
            self.pool.device.free_command_buffers(*pool, &[self.buffer]);
        }
    }
}
