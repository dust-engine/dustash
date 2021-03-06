mod dispatcher;
pub use dispatcher::{SemaphoreOp, StagedSemaphoreOp};
mod router;
pub mod semaphore;
use crate::{command::recorder::CommandExecutable, fence::Fence, Device};
use ash::{prelude::VkResult, vk};
pub use dispatcher::QueueDispatcher;
pub use router::{QueueIndex, QueueType, Queues, QueuesCreateInfo};
use std::{
    future::{Future, IntoFuture},
    sync::Arc,
};

pub struct Queue {
    pub(super) device: Arc<Device>,
    pub(super) queue: vk::Queue,
    family_index: u32,
}

impl crate::debug::DebugObject for Queue {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::QUEUE;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.queue) }
    }
}

impl crate::HasDevice for Queue {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// A thin wrapper for a Vulkan Queue. Most queue operations require host-side syncronization,
/// so these Queue operations take a mutable reference. To perform queue operations safely from multiple
/// threads, either Arc<Mutex<_>> or a threaded dispatcher would be required.
impl Queue {
    pub fn family_index(&self) -> u32 {
        self.family_index
    }
    /// A simple method to ubmit a number of independent CommandExecutables with a fence for host-size syncronization.
    /// The fence will be created and destroyed upon the completion of the job.
    #[must_use = "The submit future needs to be awaited so we can release resources upon queue completion."]
    pub fn submit_one(
        &mut self,
        commands: Vec<Arc<CommandExecutable>>,
    ) -> impl Future<Output = VkResult<()>> {
        // We always use a fence because we always need to drop used resources upon completion.
        let fence_result = Fence::new(self.device.clone(), false).and_then(|fence| unsafe {
            let buffers: Vec<vk::CommandBuffer> =
                commands.iter().map(|c| c.command_buffer.buffer).collect();
            let result = self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::builder().command_buffers(&buffers).build()],
                fence.fence,
            );
            match result {
                Ok(_) => Ok(fence),
                Err(err) => Err(err),
            }
        });

        async move {
            let fence = fence_result?;
            let result = fence.into_future().await;
            drop(commands);
            result
        }
    }

    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkQueueSubmit.html>
    pub unsafe fn submit_raw(
        &mut self,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_submit(self.queue, submits, fence)
    }

    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkQueueSubmit.html>
    pub unsafe fn submit_raw2(
        &mut self,
        submits: &[vk::SubmitInfo2],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_submit2(self.queue, submits, fence)
    }

    pub unsafe fn bind_sparse(
        &mut self,
        infos: &[vk::BindSparseInfo],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_bind_sparse(self.queue, infos, fence)
    }
}
