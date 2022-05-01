use crate::{
    frames::AcquiredFrame,
    queue::{semaphore::TimelineSemaphoreOp, SemaphoreOp},
};

mod commands;
mod host;
mod sparse_binding;
mod swapchain;

pub use commands::{CommandsFuture, CommandsStageFuture};
pub use sparse_binding::SparseBindingFuture;

/// `GPUFuture` handles the execution dependencies between queue operations.
/// It uses TimelineSemaphore for syncronization by default, but as a special case it also supports
/// waiting and signalling binary semaphores for swapchain operations.
///
/// We can use `GPUFuture::then` and `GPUFuture::then_present` to generate a render graph with futures.
/// Any render graph is essentially a [`DAG`] of queue operations, and we can represent this DAG with
/// timeline semaphores. Exactly x timeline semaphores would be required to fully represent a DAG,
/// where x is the width of the DAG.
///
/// [`DAG`]: https://en.wikipedia.org/wiki/Directed_acyclic_graph
pub trait GPUFuture {
    /// FIXME: If a future joins two parents and produces one child, the child would only have one semaphore in its pool.
    /// Ideally the child should still have two.
    /// This requires us to distribute leftover semaphores in the pool among childrens when the parent was dropped.
    /// This is easy when we only have one child, but what if we have multiple children? Should we have a global
    /// pool for the entire graph?
    fn pop_semaphore_pool(&mut self) -> TimelineSemaphoreOp;
    fn push_semaphore_pool(&mut self, semaphore: TimelineSemaphoreOp);
    /// Have the current future wait on the semaphore.
    fn wait_semaphore(&mut self, semaphore: SemaphoreOp);
    /// Signal the semaphore when the current future finish execution.
    fn signal_semaphore(&mut self, semaphore: SemaphoreOp);

    /// Returns one timeline semaphore if the future is already signalling timeline semaphores.
    fn get_one_signaled_semaphore(&self) -> Option<TimelineSemaphoreOp>;

    /// After self finish execution, do `next`.
    fn then<T: GPUFuture>(&mut self, mut next: T) -> T {
        // If self is already signalling a timeline semaphore, just have the next future wait on that.
        if let Some(existing) = self.get_one_signaled_semaphore() {
            next.wait_semaphore(existing.downgrade_arc());
            return next;
        }
        // Dequeue or create a new timeline semaphore.
        let semaphore = self.pop_semaphore_pool();
        // Signal this new semaphore when self finish execution.
        self.signal_semaphore(semaphore.clone().downgrade_arc());
        // Have the next future wait on this semaphore.
        next.wait_semaphore(semaphore.clone().downgrade_arc());
        // The next future can potentially signal the incremented semaphore.
        next.push_semaphore_pool(semaphore.increment());
        next
    }
    /// After self finish execution, present to the swapchain.
    /// Note that this does not actually call [`ash::extensions::khr::Swapchain::queue_present`]. It merely adds an execution dependency between
    /// the current future and `vkQueuePresent`. The application needs to manually call [`crate::queue::Queues::present`] each frame
    /// after all queue commands are flushed with [`crate::queue::Queues::flush`].
    fn then_present(&mut self, frame: &mut AcquiredFrame) {
        // Dequeue or create a new binary semaphore for the frame
        let binary_semaphore = frame.get_render_complete_semaphore();
        // Signal the binary semaphore when the current future finish execution
        self.signal_semaphore(SemaphoreOp {
            semaphore: binary_semaphore.clone(),
            value: 0,
        });
        // If the current future is already signalling a timeline semaphore, just use that directly.
        // Otherwise, get a new timeline semaphore and signal that on finish.
        let timeline_semaphore = if let Some(timeline_semaphore) = self.get_one_signaled_semaphore()
        {
            timeline_semaphore
        } else {
            let semaphore = self.pop_semaphore_pool();
            self.signal_semaphore(semaphore.clone().downgrade_arc());
            // The incremented semaphore is still owned by self because vkQueuePresent is not going to signal anything.
            self.push_semaphore_pool(semaphore.clone().increment());
            semaphore
        };
        // Push them into the `AcquiredFrame` so that
        // - when we call `Queues::flush`, `vkQueuePresent` will be called with the binary semaphore
        // - On the next frame, `FrameManager::acquire` will wait on the timeline semaphore
        // We need both types of semaphores because `vkQueuePresent` only supports binary semaphore,
        // and `vkWaitSemaphores` only supports timeline semaphore.
        frame
            .render_complete_semaphores
            .push((binary_semaphore, timeline_semaphore));
    }
}
