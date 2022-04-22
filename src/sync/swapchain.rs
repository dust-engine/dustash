use std::sync::Arc;

use crate::{frames::AcquiredFrame, queue::SemaphoreOp};

use super::GPUFuture;

impl AcquiredFrame {
    pub fn acquire_then<T: GPUFuture>(&self, mut future: T) -> T {
        future.wait_semaphore(SemaphoreOp {
            semaphore: self.acquire_ready_semaphore.clone(),
            value: 0,
        });
        future
    }
}
