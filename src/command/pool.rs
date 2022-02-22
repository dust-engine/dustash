use ash::{prelude::VkResult, vk};
use std::sync::Arc;

pub struct CommandPool {
    pub(crate) device: Arc<ash::Device>,
    pool: vk::CommandPool,
}

impl CommandPool {
    pub fn new(
        device: Arc<ash::Device>,
        create_info: &vk::CommandPoolCreateInfo,
    ) -> VkResult<CommandPool> {
        // Safety: No Host Syncronization rules for vkCreateCommandPool.
        unsafe {
            let pool = device.create_command_pool(create_info, None)?;
            Ok(CommandPool { device, pool })
        }
    }
    pub fn allocate_one(&mut self) -> VkResult<CommandBuffer> {
        // Safety: Host Syncronization rule for vkAllocateCommandBuffers:
        // - Host access to pAllocateInfo->commandPool must be externally synchronized.
        // We have &mut self and thus exclusive control on commandPool.
        unsafe {
            let mut buffer = vk::CommandBuffer::null();
            self.device
                .fp_v1_0()
                .allocate_command_buffers(
                    self.device.handle(),
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1)
                        .build(),
                    &mut buffer,
                )
                .result_with_success(CommandBuffer {
                    pool: self,
                    buffer,
                    owned: true,
                })
        }
    }
    pub fn allocate_n<const N: usize>(&mut self) -> VkResult<CommandBufferArray<N>> {
        // Safety: Host Syncronization rule for vkAllocateCommandBuffers:
        // - Host access to pAllocateInfo->commandPool must be externally synchronized.
        // We have &mut self and thus exclusive control on commandPool.
        unsafe {
            let mut buffers = [vk::CommandBuffer::null(); N];
            self.device
                .fp_v1_0()
                .allocate_command_buffers(
                    self.device.handle(),
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(N.try_into().unwrap())
                        .build(),
                    buffers.as_mut_ptr(),
                )
                .result_with_success(CommandBufferArray {
                    pool: self,
                    buffers,
                })
        }
    }
    pub fn allocate(&mut self, n: u32) -> VkResult<CommandBufferVec> {
        // Safety: Host Syncronization rule for vkAllocateCommandBuffers:
        // - Host access to pAllocateInfo->commandPool must be externally synchronized.
        // We have &mut self and thus exclusive control on commandPool.
        unsafe {
            let buffers = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(n)
                    .build(),
            )?;
            Ok(CommandBufferVec {
                pool: self,
                buffers,
            })
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // Safety: Host Syncronization rule for vkDestroyCommandPool:
        // - Host access to commandPool must be externally synchronized
        // We have &mut self and thus ownership on commandPool.
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

pub struct CommandBuffer<'a> {
    pub(crate) pool: &'a mut CommandPool,
    pub(crate) buffer: vk::CommandBuffer,

    // Indicates if we own the buffer.
    // When owned = true, the buffer was directly allocated from the CommandPool.
    // When owned = false, the buffer was borrowed from a CommandBufferVec or a CommandBufferArray.
    owned: bool,
}

impl<'a> Drop for CommandBuffer<'a> {
    fn drop(&mut self) {
        // Safety: Host Syncronization rule for vkFreeCommandBuffers:
        // - Host access to commandPool must be externally synchronized
        // - Host access to each member of pCommandBuffers must be externally synchronized
        // When self.owned, we have &mut self and thus ownership on self.buffer.
        // We have &mut self.pool and thus exclusive control on self.pool.pool.
        if self.owned {
            unsafe {
                self.pool
                    .device
                    .free_command_buffers(self.pool.pool, &[self.buffer]);
            }
        }
    }
}

pub struct CommandBufferArray<'a, const N: usize> {
    pool: &'a mut CommandPool,
    buffers: [vk::CommandBuffer; N],
}

impl<'a, const N: usize> CommandBufferArray<'a, N> {
    pub fn index(&mut self, n: usize) -> CommandBuffer {
        CommandBuffer {
            pool: self.pool,
            buffer: self.buffers[n],
            owned: false,
        }
    }
}

impl<'a, const N: usize> Drop for CommandBufferArray<'a, N> {
    fn drop(&mut self) {
        // Safety: Host Syncronization rule for vkFreeCommandBuffers:
        // - Host access to commandPool must be externally synchronized
        // - Host access to each member of pCommandBuffers must be externally synchronized
        // We have &mut self and thus ownership on self.buffers.
        // We have &mut self.pool and thus exclusive control on self.pool.pool.
        unsafe {
            self.pool
                .device
                .free_command_buffers(self.pool.pool, &self.buffers);
        }
    }
}

pub struct CommandBufferVec<'a> {
    pool: &'a mut CommandPool,
    buffers: Vec<vk::CommandBuffer>,
}

impl<'a> CommandBufferVec<'a> {
    pub fn index(&mut self, n: usize) -> CommandBuffer {
        CommandBuffer {
            pool: self.pool,
            buffer: self.buffers[n],
            owned: false,
        }
    }
}

impl<'a> Drop for CommandBufferVec<'a> {
    fn drop(&mut self) {
        // Safety: Host Syncronization rule for vkFreeCommandBuffers:
        // - Host access to commandPool must be externally synchronized
        // - Host access to each member of pCommandBuffers must be externally synchronized
        // We have &mut self and thus exclusive control on self.buffers.
        // We have &mut self.pool and thus exclusive control on self.pool.pool.
        unsafe {
            self.pool
                .device
                .free_command_buffers(self.pool.pool, &self.buffers);
        }
    }
}
