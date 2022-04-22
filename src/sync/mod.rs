use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use crate::{
    command::recorder::{CommandExecutable, CommandRecorder},
    frames::AcquiredFrame,
    queue::{QueueIndex, Queues, SemaphoreOp},
};

use self::{commands::CommandsFuture, semaphore::SemaphoreFuture};

mod commands;
mod semaphore;
mod sparse_binding;
mod swapchain;

pub trait GPUFuture {
    type NextFuture;
    fn pop_semaphore_pool(&mut self) -> SemaphoreOp;
    fn push_semaphore_pool(&mut self, semaphore: SemaphoreOp);
    fn wait_semaphore(&mut self, semaphore: SemaphoreOp);
    fn signal_semaphore(&mut self, semaphore: SemaphoreOp);

    /// Returns one signaled semaphore.
    fn get_one_signaled_semaphore(&self) -> Option<SemaphoreOp>;

    fn next_future(self) -> Self::NextFuture;
}
