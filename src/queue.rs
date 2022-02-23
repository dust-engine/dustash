use std::{sync::Arc, future::Future};
use ash::vk;
use crate::Device;

pub struct Queue {
    pub(super) device: Arc<Device>,
    pub(super) queue: vk::Queue,
}

struct Submission {
    // Retain reference to Device so that device may not be dropped when there are pending submissions.
    // TODO: Retain reference to Queue maybe?
    device: Arc<Device>,
    fence: vk::Fence,
}

impl Future for Submission {
    type Output = ();

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        let signaled = unsafe {
            self.device.get_fence_status(self.fence).unwrap()
        };
        if signaled {
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    }
}

pub struct SubmitInfo {
}

impl Queue {
    pub fn submit(&mut self, submits: &[SubmitInfo]) { // This should return a promise maybe
        // Safety: Host Syncronization rule for vkDestroyDevice:
        // - Host access to queue must be externally synchronized.
        // - Host access to fence must be externally synchronized.
        // We have &mut self and therefore exclusive control on queue.
        // TODO: fence syncronization
        unsafe {
            self.device.queue_submit(self.queue, submits, fence)
        }
    }
}
