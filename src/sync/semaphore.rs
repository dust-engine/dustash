use std::sync::Arc;

use ash::{
    prelude::VkResult,
    vk::{self, CommandBuffer},
};

use crate::{
    command::recorder::CommandRecorder,
    frames::AcquiredFrame,
    queue::{semaphore::Semaphore, QueueIndex, Queues, SemaphoreOp, StagedSemaphoreOp},
    sync::commands::CommandsFuture,
};

use super::GPUFuture;

pub struct SemaphoreFuture {
    pub(crate) semaphore_wait: SemaphoreOp,
}
