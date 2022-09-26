use std::{fmt::Debug, sync::Arc};

use ash::{prelude::VkResult, vk};

use crate::resources::{buffer::HasBuffer, HasImage};

use super::pool::CommandBuffer;
use crate::HasDevice;

pub struct CommandBufferBuilder {
    command_buffer: CommandBuffer,
    resource_guards: Vec<ReferencedResource>,
}

// A command buffer in Executable state.
// Once a command buffer was recorded it becomes immutable.
// Submitting a CommandExecutable in Initial state due to pool reset is a no-op.
pub struct CommandExecutable {
    pub(crate) command_buffer: CommandBuffer,
    pub(crate) _resource_guards: Vec<ReferencedResource>,
}
impl Debug for CommandExecutable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CommandExecutable")
            .field(&self.command_buffer.buffer)
            .field(&self._resource_guards.len())
            .finish()
    }
}

impl CommandExecutable {
    pub fn reset(self, release_resources: bool) -> CommandBuffer {
        let mut flags = vk::CommandBufferResetFlags::empty();
        if release_resources {
            flags |= vk::CommandBufferResetFlags::RELEASE_RESOURCES;
        }
        tracing::debug!(command_buffer = ?self.command_buffer.buffer, "Reset command buffer");
        unsafe {
            let _guard = self.command_buffer.pool.pool.lock().unwrap();
            self.command_buffer
                .pool
                .device()
                .reset_command_buffer(self.command_buffer.buffer, flags)
                .unwrap();
        }
        self.command_buffer
    }
}

pub trait CommandBufferResource: Sized + Send + Sync + 'static {
    fn command_buffer_resource(self) -> ReferencedResource {
        if std::mem::needs_drop::<Self>() {
            ReferencedResource::Boxed(Box::new(self))
        } else {
            ReferencedResource::Untracked
        }
    }
}
impl<T: Send + Sync + 'static> CommandBufferResource for Arc<T> {
    fn command_buffer_resource(self) -> ReferencedResource {
        ReferencedResource::Arc(self)
    }
}
impl<T: Send + Sync + 'static> CommandBufferResource for Box<T> {
    fn command_buffer_resource(self) -> ReferencedResource {
        ReferencedResource::Boxed(self)
    }
}
impl CommandBufferResource for Box<dyn Send + Sync> {
    fn command_buffer_resource(self) -> ReferencedResource {
        ReferencedResource::Boxed(self)
    }
}

pub enum ReferencedResource {
    Boxed(Box<dyn Send + Sync>),
    Arc(Arc<dyn Send + Sync>),
    Untracked,
}

// vk::CommandBuffer in Recording state.
// Note that during the entire lifetime of CommandRecorder, the command buffer remains in a locked state,
// so it's impossible to reset the command buffer during this time to bring this back to Initial state.
pub struct CommandRecorder<'a> {
    pub(crate) device: &'a ash::Device,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) referenced_resources: &'a mut Vec<ReferencedResource>,
}

impl CommandBuffer {
    pub fn start(self, flags: vk::CommandBufferUsageFlags) -> VkResult<CommandBufferBuilder> {
        unsafe {
            let pool = self.pool.pool.lock().unwrap();
            // Safety: Host Syncronization rule for vkBeginCommandBuffer:
            // - Host access to commandBuffer must be externally synchronized.
            // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
            // We have self and thus exclusive control on commandBuffer.
            // self.pool.pool is protected behind a mutex.
            self.pool.device().begin_command_buffer(
                self.buffer,
                &vk::CommandBufferBeginInfo::builder().flags(flags).build(),
            )?;
            drop(pool);
        }
        Ok(CommandBufferBuilder {
            command_buffer: self,
            resource_guards: Vec::new(),
        })
    }
}

impl CommandBufferBuilder {
    pub fn record<R>(&mut self, f: impl FnOnce(CommandRecorder) -> R) -> R {
        let recorder = CommandRecorder {
            device: self.command_buffer.pool.device().as_ref(),
            command_buffer: self.command_buffer.buffer,
            referenced_resources: &mut self.resource_guards,
        };
        // During recording, the cmmand pool needs to be locked too.
        let _pool = self.command_buffer.pool.pool.lock().unwrap();
        f(recorder)
    }
    pub fn end(self) -> VkResult<CommandExecutable> {
        unsafe {
            let pool = self.command_buffer.pool.pool.lock().unwrap();
            self.command_buffer
                .pool
                .device()
                .end_command_buffer(self.command_buffer.buffer)?;
            drop(pool);
            let exec = CommandExecutable {
                command_buffer: self.command_buffer,
                _resource_guards: self.resource_guards,
            };
            Ok(exec)
        }
    }
}

impl<'a> CommandRecorder<'a> {
    pub(crate) fn track_resource(&mut self, res: ReferencedResource) {
        match res {
            ReferencedResource::Untracked => (),
            _ => self.referenced_resources.push(res),
        }
    }
    pub fn copy_buffer<
        SRC: HasBuffer + CommandBufferResource,
        DST: HasBuffer + CommandBufferResource,
    >(
        &mut self,
        src_buffer: SRC,
        dst_buffer: DST,
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
                src_buffer.raw_buffer(),
                dst_buffer.raw_buffer(),
                regions,
            );
        }
        self.track_resource(src_buffer.command_buffer_resource());
        self.track_resource(dst_buffer.command_buffer_resource());
        self
    }

    pub fn copy_buffer_to_image<
        SRC: HasBuffer + CommandBufferResource,
        DST: HasImage + CommandBufferResource,
    >(
        &mut self,
        src_buffer: SRC,
        dst_image: DST,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) -> &mut Self {
        // Safety: Host Syncronization rule for vkCmdCopyBuffer:
        // - Host access to commandBuffer must be externally synchronized.
        // - Host access to the VkCommandPool that commandBuffer was allocated from must be externally synchronized.
        // We have &mut self and self.command_buffer is &mut, so we have exclusive control on self.command_buffer.buffer.
        // self.command_buffer.pool is a &mut, so we have exclusive control on self.command_buffer.pool.pool.
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                src_buffer.raw_buffer(),
                dst_image.raw_image(),
                dst_image_layout,
                regions,
            );
        }
        self.track_resource(src_buffer.command_buffer_resource());
        self.track_resource(dst_image.command_buffer_resource());
        self
    }

    pub fn clear_color_image<T: HasImage + CommandBufferResource>(
        &mut self,
        image: T,
        image_layout: vk::ImageLayout,
        clear_color_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) -> &mut Self {
        unsafe {
            self.device.cmd_clear_color_image(
                self.command_buffer,
                image.raw_image(),
                image_layout,
                clear_color_value,
                ranges,
            )
        }

        self.track_resource(image.command_buffer_resource());
        self
    }

    pub unsafe fn pipeline_barrier(
        &mut self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) -> &mut Self {
        self.device.cmd_pipeline_barrier(
            self.command_buffer,
            src_stage_mask,
            dst_stage_mask,
            dependency_flags,
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        );
        self
    }
    pub unsafe fn pipeline_barrier2(&mut self, dependency_info: &vk::DependencyInfo) -> &mut Self {
        self.device
            .cmd_pipeline_barrier2(self.command_buffer, dependency_info);
        self
    }
    pub fn bind_raytracing_pipeline(
        &mut self,
        pipeline: Arc<crate::ray_tracing::pipeline::RayTracingPipeline>,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.raw(),
            )
        }
        self.referenced_resources
            .push(pipeline.command_buffer_resource());
    }
    pub fn bind_descriptor_set(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: &crate::ray_tracing::pipeline::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                bind_point,
                pipeline_layout.raw(),
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }
    pub fn push_constants(
        &mut self,
        pipeline_layout: &crate::ray_tracing::pipeline::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                pipeline_layout.raw(),
                stage_flags,
                offset,
                constants,
            )
        }
    }
}
