use ash::prelude::VkResult;

use super::recorder::CommandRecorder;

pub struct CommandExecutable {
    pub(crate) recorder: CommandRecorder<'a>,
}

impl<'a> CommandRecorder<'a> {
    pub fn end(self) -> VkResult<CommandExecutable<'a>> {
        // Safety: Host Syncronization rule for vkEndCommandBuffer:
        // - Host access to commandBuffer must be externally synchronized.
        // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
        // We have &mut self and self.command_buffer is &mut, so we have exclusive control on self.command_buffer.buffer.
        // self.command_buffer.pool is a &mut, so we have exclusive control on self.command_buffer.pool.pool.
        unsafe {
            self.command_buffer
                .pool
                .device
                .end_command_buffer(self.command_buffer.buffer)?;
        }
        Ok(CommandExecutable { recorder: self })
    }
}
