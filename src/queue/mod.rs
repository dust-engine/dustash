mod dispatcher;
mod router;
pub mod semaphore;
pub mod timeline;
use crate::{command::recorder::CommandExecutable, fence::Fence, Device};
use ash::{prelude::VkResult, vk};
pub use router::{Queues, QueuesCreateInfo};
use std::{
    future::{Future, IntoFuture},
    sync::Arc,
};

pub struct Queue {
    pub(super) device: Arc<Device>,
    pub(super) queue: vk::Queue,
    pub(super) family_index: u32,
}

/// A thin wrapper for a Vulkan Queue. Most queue operations require host-side syncronization,
/// so these Queue operations take a mutable reference. To perform queue operations safely from multiple
/// threads, either Arc<Mutex<_>> or a threaded dispatcher would be required.
impl Queue {
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

    pub unsafe fn submit_raw(
        &mut self,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_submit(self.queue, submits, fence)
    }
    pub unsafe fn submit_raw2(
        &mut self,
        submits: &[vk::SubmitInfo2],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_submit2(self.queue, submits, fence)
    }
}
