use ash::prelude::VkResult;
use ash::vk;
use futures_util::{future::join_all, FutureExt};
use std::{future::Future, pin::Pin, sync::Arc};

use super::{
    dispatcher::{QueueDispatcher, SemaphoreOp},
    semaphore::{TimelineSemaphore, TimelineSemaphoreOp},
};
use crate::{command::recorder::CommandExecutable, Device};

type BoxedVkFuture = Pin<Box<dyn Future<Output = VkResult<()>>>>;

pub struct Timeline {
    parent_semaphore: Option<(Arc<TimelineSemaphore>, u64)>, // For branching
    semaphore: Arc<TimelineSemaphore>,
    index: u64,

    finish_task: Option<BoxedVkFuture>,
}

impl Timeline {
    pub fn new(device: Arc<Device>) -> VkResult<Self> {
        Ok(Self {
            parent_semaphore: None,
            semaphore: Arc::new(TimelineSemaphore::new(device, 0)?),
            index: 0,
            finish_task: None,
        })
    }

    /// Schedule to perform a GPU task
    ///
    /// wait_stages: Block the execution of these stages until the semaphore was signaled.
    /// Stages not specified in wait_stages can proceed before the semaphore signal operation.
    ///
    /// signal_stages: Block the semaphore signal operation on the completion of these stages.
    /// The semaphore will be signaled even if other stages are still running.
    pub fn next(
        &mut self,
        queue: &QueueDispatcher,
        executables: Vec<CommandExecutable>,
        wait_stages: vk::PipelineStageFlags2,
        signal_stages: vk::PipelineStageFlags2,
    ) -> &mut Self {
        queue.submit(
            if self.index == 0 && self.parent_semaphore.is_none() {
                // This is a brand new timeline. In the other branch, s would've been self.semaphore and
                // i would've been self.index which is 0. This is a no-op.
                Vec::new()
            } else {
                // If self.parent_semaphore is non-none, we've just branched off, so wait on the parent semaphore instead.
                let (s, i) = self.semaphore_to_wait();
                vec![SemaphoreOp {
                    semaphore: s.clone().downgrade_arc(),
                    stage_mask: wait_stages,
                    value: i,
                }]
            },
            executables,
            vec![SemaphoreOp {
                semaphore: self.semaphore.clone().downgrade_arc(),
                stage_mask: signal_stages,
                value: self.index + 1,
            }],
        );
        self.index += 1;
        self.parent_semaphore = None;
        self
    }
    /*
    TODO: For pipelined ops, there might be a need to signal multiple semaphores on multiple pipeline stages.
    We need to figure out a way to do this. For example, next() might need to return multiple branches.
    */
    /// Schedule to perform a CPU task
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
    /// wait_stages: Block the execution of these stages until the semaphore was signaled.
    /// Stages not specified in wait_stages can proceed before the semaphore signal operation.
    ///
    /// signal_stages: Block the semaphore signal operation on the completion of these stages.
    /// The semaphore will be signaled even if other stages are still running.
    ///
    /// TODO: There might be a need to block different stages of the pipeline on different timeline semaphores.
    /// We can figure out how to do this later on.
    ///
    /// TODO: There might be a need to signal different semaphores on different stages of the pipeline.
    /// We can figure out how to do this later on.
    pub fn next(
        self,
        queue: &QueueDispatcher,
        executables: Vec<CommandExecutable>,
        wait_stages: vk::PipelineStageFlags2,
        signal_stages: vk::PipelineStageFlags2,
    ) -> &'a mut Timeline {
        let (wait_semaphores, futs): (Vec<SemaphoreOp>, Vec<_>) = std::iter::once({
            let (s, i) = self.timeline.semaphore_to_wait();
            let op = SemaphoreOp {
                semaphore: s.clone().downgrade_arc(),
                stage_mask: wait_stages,
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

            let op = SemaphoreOp {
                semaphore: s.clone().downgrade_arc(),
                stage_mask: wait_stages,
                value: i,
            };
            let fut = t.finish_task;
            Some((op, fut))
        }))
        .unzip();

        let joined_fut = join_all(futs.into_iter().flatten())
            .map(|results| results.into_iter().find(|r| r.is_err()).unwrap_or(Ok(())));
        self.timeline.finish_task = Some(Box::pin(joined_fut));

        queue.submit(
            wait_semaphores,
            executables,
            vec![SemaphoreOp {
                semaphore: self.timeline.semaphore.clone().downgrade_arc(),
                stage_mask: signal_stages,
                value: self.timeline.index + 1,
            }],
        );
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
            Some((op, fut))
        }))
        .unzip();

        let joined_fut = join_all(futs.into_iter().flatten())
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
