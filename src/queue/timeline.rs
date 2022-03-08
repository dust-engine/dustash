use ash::prelude::VkResult;
use std::{future::Future, pin::Pin, sync::Arc};

use super::{semaphore::TimelineSemaphore, QueueDispatcher};
use crate::{command::recorder::CommandExecutable, Device};

pub struct TimelineSubmission {
    wait_semaphores: Vec<(Arc<TimelineSemaphore>, u64)>,
    executables: Vec<CommandExecutable>,
    signal_semaphore: (Arc<TimelineSemaphore>, u64),
}
pub struct Timeline {
    parent_semaphore: Option<(Arc<TimelineSemaphore>, u64)>, // For branching
    semaphore: Arc<TimelineSemaphore>,
    index: u64,
    finish_promise: Option<Pin<Box<dyn Future<Output = VkResult<()>>>>>,
}

impl Timeline {
    pub fn new(device: Arc<Device>) -> Timeline {
        todo!();
    }
    pub fn next(
        &mut self,
        queue: impl QueueDispatcher,
        executables: Vec<CommandExecutable>,
    ) -> &mut Self {
        let promise = queue.submit_timeline(TimelineSubmission {
            wait_semaphores: if self.index == 0 {
                Vec::new()
            } else {
                // If self.parent_semaphore is non-none, we've just branched off, so wait on the parent semaphore instead.
                if self.parent_semaphore.is_none() {
                    // Because we just branched off, self.index is gotta be 0. Waiting on self.semaphore would have been no-op.
                    assert_eq!(self.index, 0);
                }
                let s = self
                    .parent_semaphore
                    .as_ref()
                    .map(|(s, _)| s)
                    .unwrap_or(&self.semaphore)
                    .clone();
                let i = self
                    .parent_semaphore
                    .as_ref()
                    .map(|(s, i)| *i)
                    .unwrap_or(self.index);
                vec![(s, i)]
            },
            executables,
            signal_semaphore: (self.semaphore.clone(), self.index + 1),
        });
        self.finish_promise = Some(match self.finish_promise.take() {
            None => promise,
            Some(p) => Box::pin(async {
                p.await?;
                promise.await?;
                Ok(())
            }),
        });
        self.index += 1;
        self
    }
    pub fn join_n<const N: usize>(&mut self, timelines: [Timeline; N]) -> TimelineJoin<N> {
        debug_assert_ne!(
            self.index, 0,
            "Use the join method on one of the timelines instead."
        );
        TimelineJoin {
            timeline: self,
            timelines,
        }
    }
    pub fn finish(mut self) -> impl Future<Output = VkResult<()>> {
        self.finish_promise
            .take()
            .expect("Cannot finish an empty timeline.")
    }
    pub fn branch(&self) -> Self {
        assert!(
            self.parent_semaphore.is_none(),
            "Can't branch on an empty branch"
        );
        Self {
            parent_semaphore: Some((self.semaphore.clone(), self.index)),
            semaphore: Arc::new(TimelineSemaphore::new(self.semaphore.device.clone(), 0).unwrap()),
            index: 0,
            finish_promise: None,
        }
    }
}
pub struct TimelineJoin<'a, const N: usize> {
    timeline: &'a mut Timeline,
    timelines: [Timeline; N],
}
impl<'a, const N: usize> TimelineJoin<'a, N> {
    pub fn next(
        self,
        queue: impl QueueDispatcher,
        executables: Vec<CommandExecutable>,
    ) -> &'a mut Timeline {
        let mut wait_semaphores: Vec<(Arc<TimelineSemaphore>, u64)> = Vec::with_capacity(N + 1);
        wait_semaphores.push((self.timeline.semaphore.clone(), self.timeline.index));
        for t in self.timelines.into_iter() {
            if t.parent_semaphore.is_none() {
                // Because we just branched off, self.index is gotta be 0. Waiting on self.semaphore would have been no-op.
                assert_eq!(t.index, 0);
            }
            let s = t
                .parent_semaphore
                .as_ref()
                .map(|(s, _)| s)
                .unwrap_or(&t.semaphore)
                .clone();
            let i = t
                .parent_semaphore
                .as_ref()
                .map(|(_, i)| *i)
                .unwrap_or(t.index);
            wait_semaphores.push((s, i));
        }

        let promise = queue.submit_timeline(TimelineSubmission {
            wait_semaphores,
            executables,
            signal_semaphore: (self.timeline.semaphore.clone(), self.timeline.index + 1),
        });

        self.timeline.finish_promise = Some(match self.timeline.finish_promise.take() {
            None => promise,
            Some(p) => Box::pin(async {
                p.await?;
                promise.await?;
                Ok(())
            }),
        });
        self.timeline.index += 1;
        self.timeline
    }
}
