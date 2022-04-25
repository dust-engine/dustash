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
    type NextFuture;
    fn pop_semaphore_pool(&mut self) -> TimelineSemaphoreOp;
    fn push_semaphore_pool(&mut self, semaphore: TimelineSemaphoreOp);
    fn wait_semaphore(&mut self, semaphore: SemaphoreOp);
    fn signal_semaphore(&mut self, semaphore: SemaphoreOp);

    /// Returns one signaled semaphore.
    fn get_one_signaled_semaphore(&self) -> Option<TimelineSemaphoreOp>;

    fn next_future(self) -> Self::NextFuture;

    fn then<T: GPUFuture>(&mut self, mut next: T) -> T::NextFuture {
        if let Some(existing) = self.get_one_signaled_semaphore() {
            next.wait_semaphore(existing.downgrade_arc());
            return next.next_future();
        }
        let semaphore = self.pop_semaphore_pool();
        self.signal_semaphore(semaphore.clone().downgrade_arc());
        next.wait_semaphore(semaphore.clone().downgrade_arc());
        next.push_semaphore_pool(semaphore.increment());
        next.next_future()
    }
    fn then_present(&mut self, frame: &mut AcquiredFrame) {
        let binary_semaphore = frame.get_render_complete_semaphore();
        self.signal_semaphore(SemaphoreOp {
            semaphore: binary_semaphore.clone(),
            value: 0,
        });
        let timeline_semaphore = if let Some(timeline_semaphore) = self.get_one_signaled_semaphore()
        {
            timeline_semaphore
        } else {
            let semaphore = self.pop_semaphore_pool();
            self.signal_semaphore(semaphore.clone().downgrade_arc());
            semaphore
        };
        frame
            .render_complete_semaphores
            .push((binary_semaphore, timeline_semaphore));
    }
}
