use std::sync::Arc;

use crate::queue::{
    semaphore::{TimelineSemaphore, TimelineSemaphoreOp},
    QueueIndex, Queues, SemaphoreOp,
};
use ash::vk;

use super::GPUFuture;
use crate::HasDevice;
pub struct SparseBindingFuture<'q> {
    queues: &'q Queues,
    pub(crate) queue: QueueIndex,
    pub(crate) available_semaphore_pool: Vec<TimelineSemaphoreOp>,
    pub(crate) semaphore_waits: Vec<SemaphoreOp>,
    pub(crate) semaphore_signals: Vec<SemaphoreOp>,

    buffer_binds: Vec<(vk::Buffer, Box<[vk::SparseMemoryBind]>)>,
    image_opaque_binds: Vec<(vk::Image, Box<[vk::SparseMemoryBind]>)>,
    image_binds: Vec<(vk::Image, Box<[vk::SparseImageMemoryBind]>)>,
}

impl<'q> GPUFuture for SparseBindingFuture<'q> {
    fn pop_semaphore_pool(&mut self) -> TimelineSemaphoreOp {
        self.available_semaphore_pool.pop().unwrap_or_else(|| {
            let semaphore =
                TimelineSemaphore::new(self.queues.of_index(self.queue).device().clone(), 0)
                    .unwrap();
            let semaphore = Arc::new(semaphore);
            TimelineSemaphoreOp {
                semaphore: semaphore,
                value: 1,
            }
        })
    }
    fn push_semaphore_pool(&mut self, semaphore: TimelineSemaphoreOp) {
        self.available_semaphore_pool.push(semaphore);
    }
    fn wait_semaphore(&mut self, semaphore: SemaphoreOp) {
        self.semaphore_waits.push(semaphore);
    }
    fn signal_semaphore(&mut self, semaphore: SemaphoreOp) {
        self.semaphore_signals.push(semaphore);
    }

    /// Returns one signaled semaphore.
    fn get_one_signaled_semaphore(&self) -> Option<TimelineSemaphoreOp> {
        self.semaphore_signals
            .iter()
            .find(|&s| s.is_timeline())
            .map(|s| s.clone().as_timeline())
    }
}

impl<'q> SparseBindingFuture<'q> {
    pub fn bind_buffer(mut self, buffer: vk::Buffer, binds: Box<[vk::SparseMemoryBind]>) -> Self {
        self.buffer_binds.push((buffer, binds));
        self
    }
    pub fn bind_image_opaque(
        mut self,
        image: vk::Image,
        binds: Box<[vk::SparseMemoryBind]>,
    ) -> Self {
        self.image_opaque_binds.push((image, binds));
        self
    }
    pub fn bind_image(mut self, image: vk::Image, binds: Box<[vk::SparseImageMemoryBind]>) -> Self {
        self.image_binds.push((image, binds));
        self
    }
}

impl Drop for SparseBindingFuture<'_> {
    fn drop(&mut self) {
        use std::mem::take;
        // When dropping CommandsFuture it is no longer possible to add semaphores to it.
        // Therefore, this is in fact the best opportunity to flush.
        self.queues.of_index(self.queue).sparse_bind(
            take(&mut self.semaphore_waits).into_boxed_slice(),
            take(&mut self.buffer_binds).into_boxed_slice(),
            take(&mut self.image_opaque_binds).into_boxed_slice(),
            take(&mut self.image_binds).into_boxed_slice(),
            take(&mut self.semaphore_signals).into_boxed_slice(),
        );
    }
}
