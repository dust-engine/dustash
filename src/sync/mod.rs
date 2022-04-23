use crate::{frames::AcquiredFrame, queue::SemaphoreOp};

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
    fn pop_semaphore_pool(&mut self) -> SemaphoreOp;
    fn push_semaphore_pool(&mut self, semaphore: SemaphoreOp);
    fn wait_semaphore(&mut self, semaphore: SemaphoreOp);
    fn signal_semaphore(&mut self, semaphore: SemaphoreOp);

    /// Returns one signaled semaphore.
    fn get_one_signaled_semaphore(&self) -> Option<SemaphoreOp>;

    fn next_future(self) -> Self::NextFuture;

    fn then<T: GPUFuture>(&mut self, mut next: T) -> T::NextFuture {
        if let Some(existing) = self.get_one_signaled_semaphore() {
            next.wait_semaphore(existing);
            return next.next_future();
        }
        let semaphore = self.pop_semaphore_pool();
        self.signal_semaphore(semaphore.clone());
        next.wait_semaphore(semaphore.clone());
        next.push_semaphore_pool(semaphore.increment());
        next.next_future()
    }
    fn then_present(&mut self, frame: &mut AcquiredFrame) {
        self.signal_semaphore(SemaphoreOp {
            semaphore: frame.render_complete_semaphore(),
            value: 0,
        });
    }
}
