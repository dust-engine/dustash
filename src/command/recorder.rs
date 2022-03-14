use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use crate::resources;

use super::pool::CommandBuffer;

// A command buffer in Executable state.
// Once a command buffer was recorded it becomes immutable.
// Submitting a CommandExecutable in Initial state due to pool reset is a no-op.
pub struct CommandExecutable {
    pub(crate) command_buffer: CommandBuffer,
    pub(crate) _resource_guards: Vec<Arc<dyn Send + Sync>>,
}

// vk::CommandBuffer in Recording state.
// Note that during the entire lifetime of CommandRecorder, the command buffer remains in a locked state,
// so it's impossible to reset the command buffer during this time to bring this back to Initial state.
pub struct CommandRecorder<'a> {
    device: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
    referenced_resources: Vec<Arc<dyn Send + Sync>>,
}

impl CommandBuffer {
    pub fn record(
        self,
        flags: vk::CommandBufferUsageFlags,
        record: impl FnOnce(&mut CommandRecorder),
    ) -> VkResult<CommandExecutable> {
        let pool = self.pool.pool.lock();
        let referenced_resources = unsafe {
            // Safety: Host Syncronization rule for vkBeginCommandBuffer:
            // - Host access to commandBuffer must be externally synchronized.
            // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
            // We have self and thus exclusive control on commandBuffer.
            // self.pool.pool is protected behind a mutex.
            self.pool.device.begin_command_buffer(
                self.buffer,
                &vk::CommandBufferBeginInfo::builder().flags(flags).build(),
            )?;
            let mut recorder = CommandRecorder {
                device: self.pool.device.as_ref(),
                command_buffer: self.buffer,
                referenced_resources: Vec::new(),
            };
            record(&mut recorder);

            // Safety: Host Syncronization rule for vkBeginCommandBuffer:
            // - Host access to commandBuffer must be externally synchronized.
            // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
            // We have self and thus exclusive control on commandBuffer.
            // self.pool.pool is protected behind a mutex.
            self.pool.device.end_command_buffer(self.buffer)?;
            recorder.referenced_resources
        };
        drop(pool);
        Ok(CommandExecutable {
            command_buffer: self,
            _resource_guards: referenced_resources,
        })
    }
}

impl<'a> CommandRecorder<'a> {
    pub fn copy_buffer(
        &mut self,
        src_buffer: Arc<resources::Buffer>,
        dst_buffer: Arc<resources::Buffer>,
        regions: &[vk::BufferCopy],
    ) -> &mut Self {
        // Safety: Host Syncronization rule for vkCmdCopyBuffer:
        // - Host access to commandBuffer must be externally synchronized.
        // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
        // We have &mut self and self.command_buffer is &mut, so we have exclusive control on self.command_buffer.buffer.
        // self.command_buffer.pool is a &mut, so we have exclusive control on self.command_buffer.pool.pool.
        unsafe {
            self.device.cmd_copy_buffer(
                self.command_buffer,
                src_buffer.buffer,
                dst_buffer.buffer,
                regions,
            );
        }
        self.referenced_resources.push(src_buffer);
        self.referenced_resources.push(dst_buffer);
        self
    }
}
