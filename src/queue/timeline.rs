use ash::prelude::VkResult;
use ash::vk;
use futures_util::{future::join_all, FutureExt};
use std::{future::Future, pin::Pin, process::Output, sync::Arc};

use super::{
    dispatcher::{QueueDispatcher, SemaphoreOp, Submission},
    semaphore::{TimelineSemaphore, TimelineSemaphoreOp},
};
use crate::{command::recorder::CommandExecutable, queue::semaphore::Semaphore, Device};

type BoxedVkFuture = Pin<Box<dyn Future<Output = VkResult<()>>>>;

pub struct Timeline {
    parent_semaphore: Option<(Arc<TimelineSemaphore>, u64)>, // For branching
    semaphore: Arc<TimelineSemaphore>,
    index: u64,

    finish_task: Option<BoxedVkFuture>,
}

impl Timeline {
    pub fn new(device: Arc<Device>) -> Timeline {
        todo!();
    }
    pub fn next(
        &mut self,
        queue: &QueueDispatcher,
        executables: Vec<CommandExecutable>,
    ) -> &mut Self {
        queue.submit(Submission {
            wait_semaphores: if self.index == 0 && self.parent_semaphore.is_none() {
                // This is a brand new timeline. In the other branch, s would've been self.semaphore and
                // i would've been self.index which is 0. This is a no-op.
                Vec::new()
            } else {
                // If self.parent_semaphore is non-none, we've just branched off, so wait on the parent semaphore instead.
                let (s, i) = self.semaphore_to_wait();
                vec![SemaphoreOp {
                    semaphore: s.clone().downgrade_arc(),
                    stage_mask: vk::PipelineStageFlags2::empty(), // TODO
                    value: i,
                }]
            },
            executables,
            signal_semaphore: vec![SemaphoreOp {
                semaphore: self.semaphore.clone().downgrade_arc(),
                stage_mask: vk::PipelineStageFlags2::empty(), // TODO
                value: self.index + 1,
            }],
        });
        self.index += 1;
        self.parent_semaphore = None;
        self
    }
    pub fn then(&mut self, task: impl Future<Output = VkResult<()>> + 'static) -> &mut Self {
        let semaphore_to_wait = if self.index == 0 && self.parent_semaphore.is_none() {
            None
        } else {
            let (semaphore, index) = self.semaphore_to_wait();
            Some(TimelineSemaphoreOp {
                semaphore: semaphore.clone(),
                value: index,
            })
        };

        let semaphore_to_signal = TimelineSemaphoreOp {
            semaphore: self.semaphore.clone(),
            value: self.index + 1,
        };
        let prev_task = self.finish_task.take();
        self.finish_task = Some(Box::pin(async {
            if let Some(task) = prev_task {
                task.await?;
            }
            if let Some(semaphore_to_wait) = semaphore_to_wait {
                semaphore_to_wait.wait().await?;
            }
            task.await?;
            semaphore_to_signal.wait().await?;
            Ok(())
        }));
        self.index += 1;
        self.parent_semaphore = None;
        self
    }
    pub fn join(&mut self, timelines: Vec<Timeline>) -> TimelineJoin {
        debug_assert_ne!(
            self.index, 0,
            "Use the join method on one of the timelines instead."
        );
        TimelineJoin {
            timeline: self,
            timelines,
        }
    }
    pub async fn finish(mut self) -> VkResult<()> {
        if let Some(task) = self.finish_task.take() {
            task.await?;
        }
        self.semaphore.wait(self.index).await?;
        Ok(())
    }
    pub fn branch(&mut self) -> Self {
        assert!(
            self.parent_semaphore.is_none(),
            "Can't branch on an empty branch"
        );
        let finish_task = self.finish_task.take().map(|p| p.shared());
        let finish_task_other = finish_task.clone();
        self.finish_task = finish_task.map(|f| Box::pin(f) as _);
        Self {
            parent_semaphore: Some((self.semaphore.clone(), self.index)),
            semaphore: Arc::new(
                TimelineSemaphore::new(self.semaphore.device().clone(), 0).unwrap(),
            ),
            index: 0,
            finish_task: finish_task_other.map(|f| Box::pin(f) as _),
        }
    }
    // Returns the semaphore to wait, with branching considered.
    // If we just branched off another timeline, wait on the semaphore on the other timeline instead.
    // Otherwise, wait on the semaphore owned by the current timeline.
    fn semaphore_to_wait(&self) -> (&Arc<TimelineSemaphore>, u64) {
        match self.parent_semaphore.as_ref() {
            Some((parent_semaphore, value)) => (parent_semaphore, *value),
            None => (&self.semaphore, self.index),
        }
    }
}
pub struct TimelineJoin<'a> {
    timeline: &'a mut Timeline,
    timelines: Vec<Timeline>,
}
impl<'a> TimelineJoin<'a> {
    pub fn next(
        self,
        queue: &QueueDispatcher,
        executables: Vec<CommandExecutable>,
    ) -> &'a mut Timeline {
        let (wait_semaphores, futs): (Vec<SemaphoreOp>, Vec<_>) = std::iter::once({
            let (s, i) = self.timeline.semaphore_to_wait();
            let op = SemaphoreOp {
                semaphore: s.clone().downgrade_arc(),
                stage_mask: vk::PipelineStageFlags2::empty(),
                value: i,
            };
            let fut = self.timeline.finish_task.take();
            (op, fut)
        })
        .chain(self.timelines.into_iter().filter_map(|mut t| {
            if t.parent_semaphore.is_none() && t.index == 0 {
                return None;
            }
            let (s, i) = t.semaphore_to_wait();

            let op = SemaphoreOp {
                semaphore: s.clone().downgrade_arc(),
                stage_mask: vk::PipelineStageFlags2::empty(),
                value: i,
            };
            let fut = t.finish_task;
            return Some((op, fut));
        }))
        .unzip();

        let joined_fut = join_all(futs.into_iter().filter_map(|x| x))
            .map(|results| results.into_iter().find(|r| r.is_err()).unwrap_or(Ok(())));
        self.timeline.finish_task = Some(Box::pin(joined_fut));

        queue.submit(Submission {
            wait_semaphores,
            executables,
            signal_semaphore: vec![SemaphoreOp {
                semaphore: self.timeline.semaphore.clone().downgrade_arc(),
                stage_mask: vk::PipelineStageFlags2::empty(),
                value: self.timeline.index + 1,
            }],
        });
        self.timeline.index += 1;
        self.timeline.parent_semaphore = None;

        self.timeline
    }
    pub fn then(self, future: impl Future<Output = VkResult<()>> + 'static) -> &'a mut Timeline {
        let (wait_semaphores, futs): (Vec<TimelineSemaphoreOp>, Vec<_>) = std::iter::once({
            let (s, i) = self.timeline.semaphore_to_wait();
            let op = TimelineSemaphoreOp {
                semaphore: s.clone(),
                value: i,
            };
            let fut = self.timeline.finish_task.take();
            (op, fut)
        })
        .chain(self.timelines.into_iter().filter_map(|t| {
            if t.parent_semaphore.is_none() && t.index == 0 {
                return None;
            }
            let (s, i) = t.semaphore_to_wait();

            let op = TimelineSemaphoreOp {
                semaphore: s.clone(),
                value: i,
            };
            let fut = t.finish_task;
            return Some((op, fut));
        }))
        .unzip();

        let joined_fut = join_all(futs.into_iter().filter_map(|x| x))
            .map(|results| results.into_iter().find(|r| r.is_err()).unwrap_or(Ok(())));
        let fut = async {
            joined_fut.await?;
            TimelineSemaphore::wait_many(wait_semaphores).await?;
            future.await?;
            Ok(())
        };
        self.timeline.finish_task = Some(Box::pin(fut));
        self.timeline.index += 1;
        self.timeline.parent_semaphore = None;
        self.timeline
    }
}

// TODO: Mask
// TODO: unblock timing. only start a new threaded when needed.
// TODO: tests.
// TOOD: async optimization. then on CPU task shouldn't need to increment.
