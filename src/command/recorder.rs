use ash::{prelude::VkResult, vk};

use super::pool::CommandBuffer;

// vk::CommandBuffer in Recording state.
pub struct CommandRecorder<'a> {
    pub(crate) command_buffer: CommandBuffer<'a>,
}

impl<'a> CommandBuffer<'a> {
    pub fn begin(self, flags: vk::CommandBufferUsageFlags) -> VkResult<CommandRecorder<'a>> {
        // Safety: Host Syncronization rule for vkBeginCommandBuffer:
        // - Host access to commandBuffer must be externally synchronized.
        // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
        // We have &mut self and thus exclusive control on commandBuffer.
        // self.pool is a &mut, so we have exclusive control on VkCommandPool.
        unsafe {
            self.pool.device.begin_command_buffer(
                self.buffer,
                &vk::CommandBufferBeginInfo::builder().flags(flags).build(),
            )?;
            Ok(CommandRecorder {
                command_buffer: self,
            })
        }
    }
}

impl<'a> CommandRecorder<'a> {
    pub fn copy_buffer(
        &mut self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) -> &mut Self {
        // Safety: Host Syncronization rule for vkCmdCopyBuffer:
        // - Host access to commandBuffer must be externally synchronized.
        // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
        // We have &mut self and self.command_buffer is &mut, so we have exclusive control on self.command_buffer.buffer.
        // self.command_buffer.pool is a &mut, so we have exclusive control on self.command_buffer.pool.pool.
        unsafe {
            self.command_buffer.pool.device.cmd_copy_buffer(
                self.command_buffer.buffer,
                src_buffer,
                dst_buffer,
                regions,
            );
        }
        self
    }
}
