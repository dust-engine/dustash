use crate::{frames::AcquiredFrame, queue::SemaphoreOp};

mod commands;
mod sparse_binding;
mod swapchain;

pub use commands::{CommandsFuture, CommandsStageFuture};
pub use sparse_binding::SparseBindingFuture;

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
