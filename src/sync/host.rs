use std::{future::IntoFuture, sync::Arc};

use std::future::Future;

use crate::{
    queue::{
        semaphore::{TimelineSemaphore, TimelineSemaphoreOp},
        SemaphoreOp,
    },
    Device,
};

use super::GPUFuture;

#[must_use]
pub struct HostFuture<T: Future> {
    device: Arc<Device>,
    pub(crate) available_semaphore_pool: Vec<SemaphoreOp>,
    pub(crate) semaphore_waits: Vec<TimelineSemaphoreOp>,
    pub(crate) semaphore_signals: Vec<TimelineSemaphoreOp>,

    future: Option<T>,
}

impl<T: Future> IntoFuture for HostFuture<T> {
    type Output = ();
    type IntoFuture = impl Future<Output = ()>;

    fn into_future(mut self) -> Self::IntoFuture {
        use std::mem::take;
        let future = self.future.take().unwrap();
        let waits = take(&mut self.semaphore_waits);
        let signals = take(&mut self.semaphore_signals);
        async {
            TimelineSemaphore::wait_many(waits).await.unwrap();
            future.await;
            for signal in signals {
                signal.signal();
            }
        }
    }
}

impl<T: Future> GPUFuture for HostFuture<T> {
    type NextFuture = Self;

    fn pop_semaphore_pool(&mut self) -> SemaphoreOp {
        self.available_semaphore_pool.pop().unwrap_or_else(|| {
            let semaphore = TimelineSemaphore::new(self.device.clone(), 0).unwrap();
            let semaphore = Arc::new(semaphore);
            SemaphoreOp {
                semaphore: semaphore.downgrade_arc(),
                value: 1,
            }
        })
    }

    fn push_semaphore_pool(&mut self, semaphore: SemaphoreOp) {
        self.available_semaphore_pool.push(semaphore);
    }

    fn wait_semaphore(&mut self, semaphore: SemaphoreOp) {
        assert!(semaphore.is_timeline());
        self.semaphore_waits.push(semaphore.as_timeline());
    }

    fn signal_semaphore(&mut self, semaphore: SemaphoreOp) {
        assert!(semaphore.is_timeline());
        self.semaphore_signals.push(semaphore.as_timeline());
    }

    fn get_one_signaled_semaphore(&self) -> Option<SemaphoreOp> {
        self.semaphore_signals
            .get(0)
            .map(|s| s.clone().downgrade_arc())
    }

    fn next_future(self) -> Self::NextFuture {
        self
    }
}
