use std::{
    future::{Future, IntoFuture},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use ash::{prelude::VkResult, vk};
use futures::FutureExt;

use crate::{command::recorder::CommandExecutable, fence::Fence};

use self::submission::TimelineSubmission;

use super::{
    semaphore::{Semaphore, TimelineSemaphore},
    Queue,
};

/// Queue operations require exclusive access to the Queue object, and it's usually more
/// performant to batch queue submissions together.
/// The QueueDispatcher provides shared references to a Queue, and acts like a buffer
/// so that the queue operations can be submitted in batch on a frame-to-frame basis.
pub struct QueueDispatcher {
    queue: Queue,
    submissions: crossbeam::queue::SegQueue<QueueSubmission>,
    submission_count: AtomicUsize,

    // This fence gets replaced with a new fence every time we flush.
    // It gets signaled whenever the flush finishes.
    fence_task: futures::future::Shared<blocking::Task<VkResult<()>>>,
    fence_raw: vk::Fence,
}

impl QueueDispatcher {
    pub fn new(queue: Queue) -> Self {
        QueueDispatcher {
            queue,
            submissions: crossbeam::queue::SegQueue::new(),
            submission_count: AtomicUsize::new(0),
            fence_task: todo!(),
            fence_raw: vk::Fence::null(),
        }
    }
    /// Returns a future that resolves after flush() when all submissions in the batch completes.
    pub fn submit(
        &self,
        submission: submission::FencedSubmission,
    ) -> impl Future<Output = VkResult<()>> {
        // submission actually borrows the FencedSubmission here by taking the raw handles of semaphores and fences.
        // We need to ensure that FencedSubmission outlives borrowed_submission.
        let borrowed_submission = QueueSubmission {
            wait_semaphores: submission
                .wait_semaphores
                .iter()
                .map(|waits| vk::SemaphoreSubmitInfo {
                    semaphore: waits.semaphore.semaphore,
                    stage_mask: waits.stage_mask,
                    ..Default::default()
                })
                .collect(),
            signal_semaphores: submission
                .signal_semaphore
                .iter()
                .map(|signals| vk::SemaphoreSubmitInfo {
                    semaphore: signals.semaphore.semaphore,
                    stage_mask: signals.stage_mask,
                    ..Default::default()
                })
                .collect(),
            command_buffers: submission
                .executables
                .iter()
                .map(|exe| vk::CommandBufferSubmitInfo {
                    command_buffer: exe.command_buffer.buffer,
                    ..Default::default()
                })
                .collect(),
        };

        self.submission_count.fetch_add(1, Ordering::Relaxed);
        self.submissions.push(borrowed_submission);

        let fence = self.fence_task.clone();
        async move {
            fence.await?;
            // borrowed_submission lives as long as the queue submission is still pending.
            // When the queue finishes, it releases borrowed_submission, so we're free to drop submission now.
            drop(submission);
            Ok(())
        }
    }

    /// Returns a future that resolves after flush() when this particular submission finishes.
    pub fn submit_timeline(
        &self,
        submission: submission::TimelineSubmission,
    ) -> impl Future<Output = VkResult<()>> {
        // submission actually borrows the FencedSubmission here by taking the raw handles of semaphores and fences.
        // We need to ensure that FencedSubmission outlives borrowed_submission.
        let borrowed_submission = QueueSubmission {
            wait_semaphores: submission
                .wait_semaphores
                .iter()
                .map(|waits| vk::SemaphoreSubmitInfo {
                    semaphore: waits.semaphore.semaphore,
                    stage_mask: waits.stage_mask,
                    value: waits.value,
                    ..Default::default()
                })
                .collect(),
            signal_semaphores: vec![vk::SemaphoreSubmitInfo {
                semaphore: submission.signal_semaphore.semaphore.semaphore,
                stage_mask: submission.signal_semaphore.stage_mask,
                ..Default::default()
            }],
            command_buffers: submission
                .executables
                .iter()
                .map(|exe| vk::CommandBufferSubmitInfo {
                    command_buffer: exe.command_buffer.buffer,
                    ..Default::default()
                })
                .collect(),
        };

        self.submission_count.fetch_add(1, Ordering::Relaxed);
        self.submissions.push(borrowed_submission);

        // TODO: wait on timeline semaphore instead.
        let fence = self.fence_task.clone();
        async move {
            fence.await?;
            drop(submission);
            Ok(())
        }
    }

    pub fn flush(&mut self) -> VkResult<()> {
        let num_submissions = *self.submission_count.get_mut();
        *self.submission_count.get_mut() = 0;
        let mut submission_infos: Vec<vk::SubmitInfo2> = Vec::with_capacity(num_submissions);
        {
            let mut i: usize = 0;
            loop {
                let op = match self.submissions.pop() {
                    Some(op) => op,
                    None => break,
                };
                submission_infos.push(vk::SubmitInfo2 {
                    wait_semaphore_info_count: op.wait_semaphores.len() as u32,
                    p_wait_semaphore_infos: op.wait_semaphores.as_ptr(),
                    command_buffer_info_count: op.command_buffers.len() as u32,
                    p_command_buffer_infos: op.command_buffers.as_ptr(),
                    signal_semaphore_info_count: op.wait_semaphores.len() as u32,
                    p_signal_semaphore_infos: op.wait_semaphores.as_ptr(),
                    ..Default::default()
                });
                i += 1;
            }
            assert_eq!(i, num_submissions);
        }
        unsafe {
            self.queue
                .submit_raw2(submission_infos.as_slice(), self.fence_raw)?;
        }

        let new_fence = Fence::new(self.queue.device.clone(), false)?;
        self.fence_raw = new_fence.fence;
        self.fence_task = new_fence.into_future().shared();
        Ok(())
    }
}

pub mod submission {
    use super::*;
    pub struct BinarySemaphoreOp {
        pub semaphore: Arc<Semaphore>,
        pub stage_mask: vk::PipelineStageFlags2,
    }

    pub struct TimelineSemaphoreOp {
        pub semaphore: Arc<Semaphore>,
        pub stage_mask: vk::PipelineStageFlags2,
        pub value: u64,
    }

    pub struct FencedSubmission {
        pub wait_semaphores: Vec<BinarySemaphoreOp>,
        pub executables: Vec<CommandExecutable>,
        pub signal_semaphore: Vec<BinarySemaphoreOp>,
    }

    pub struct TimelineSubmission {
        pub wait_semaphores: Vec<TimelineSemaphoreOp>,
        pub executables: Vec<CommandExecutable>,
        pub signal_semaphore: TimelineSemaphoreOp,
    }
}

struct QueueSubmission {
    wait_semaphores: Vec<vk::SemaphoreSubmitInfo>,
    signal_semaphores: Vec<vk::SemaphoreSubmitInfo>,
    command_buffers: Vec<vk::CommandBufferSubmitInfo>,
}
