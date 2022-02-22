use ash::{vk, prelude::VkResult};

use super::pool::CommandBuffer;

pub struct CommandRecorder {}

impl<'a> CommandBuffer<'a> {
    pub fn begin(&mut self, flags: vk::CommandBufferUsageFlags) -> VkResult<()> {
        // Safety: Host Syncronization rule for vkBeginCommandBuffer:
        // - Host access to commandBuffer must be externally synchronized.
        // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
        // We have &mut self and thus exclusive control on commandBuffer.
        // self.pool is a &mut, so we have exclusive control on VkCommandPool.
        unsafe {
            self.pool.device.begin_command_buffer(
                self.buffer,
                &vk::CommandBufferBeginInfo::builder().flags(flags).build(),
            )
        }
    }
}
