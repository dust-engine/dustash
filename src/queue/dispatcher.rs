use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use ash::{prelude::VkResult, vk};

use crate::{command::recorder::CommandExecutable, fence::Fence};

use super::{semaphore::Semaphore, Queue, QueueType};

/// Queue operations require exclusive access to the Queue object, and it's usually more
/// performant to batch queue submissions together.
/// The QueueDispatcher provides shared references to a Queue, and acts like a buffer
/// so that the queue operations can be submitted in batch on a frame-to-frame basis.
pub struct QueueDispatcher {
    pub(crate) queue: Queue,
    assigned_queue_type: Option<QueueType>,
    submissions: crossbeam::queue::SegQueue<Submission>,
    submission_count: AtomicUsize,
}

impl QueueDispatcher {
    pub fn new(queue: Queue, assigned_type: Option<QueueType>) -> Self {
        QueueDispatcher {
            queue,
            assigned_queue_type: assigned_type,
            submissions: crossbeam::queue::SegQueue::new(),
            submission_count: AtomicUsize::new(0),
        }
    }
    pub fn family_index(&self) -> u32 {
        self.queue.family_index
    }
    /// Returns a future that resolves after flush() when all submissions in the batch completes.
    pub fn submit(&self, submission: Submission) {
        self.submission_count.fetch_add(1, Ordering::Relaxed);
        self.submissions.push(submission);
    }

    // TODO: bind_sparse

    pub fn flush(&mut self) -> VkResult<Option<QueueSubmissionFence>> {
        let num_submissions = *self.submission_count.get_mut();
        if num_submissions == 0 {
            return Ok(None);
        }
        *self.submission_count.get_mut() = 0;
        let mut submissions: Vec<Submission> = Vec::with_capacity(num_submissions);
        let mut wait_semaphore_count: usize = 0;
        let mut signal_semaphore_count: usize = 0;
        let mut executables_count: usize = 0;
        {
            while let Some(op) = self.submissions.pop() {
                wait_semaphore_count += op.wait_semaphores.len();
                signal_semaphore_count += op.signal_semaphore.len();
                executables_count += op.executables.len();
                submissions.push(op);
            }
            assert_eq!(submissions.len(), num_submissions);
        }

        let mut wait_semaphores: Vec<vk::SemaphoreSubmitInfo> =
            Vec::with_capacity(wait_semaphore_count);
        let mut signal_semaphores: Vec<vk::SemaphoreSubmitInfo> =
            Vec::with_capacity(signal_semaphore_count);
        let mut command_buffers: Vec<vk::CommandBufferSubmitInfo> =
            Vec::with_capacity(executables_count);
        let mut submit_infos: Vec<vk::SubmitInfo2> = Vec::with_capacity(num_submissions);

        for submission in submissions.iter() {
            unsafe {
                submit_infos.push(vk::SubmitInfo2 {
                    wait_semaphore_info_count: submission.wait_semaphores.len() as u32,
                    p_wait_semaphore_infos: wait_semaphores.as_ptr().add(wait_semaphores.len()),
                    command_buffer_info_count: submission.executables.len() as u32,
                    p_command_buffer_infos: command_buffers.as_ptr().add(command_buffers.len()),
                    signal_semaphore_info_count: submission.signal_semaphore.len() as u32,
                    p_signal_semaphore_infos: signal_semaphores
                        .as_ptr()
                        .add(signal_semaphores.len()),
                    ..Default::default()
                });
            }
            for wait in submission.wait_semaphores.iter() {
                wait_semaphores.push(vk::SemaphoreSubmitInfo {
                    semaphore: wait.semaphore.semaphore,
                    value: wait.value,
                    stage_mask: wait.stage_mask,
                    ..Default::default()
                });
            }
            for signal in submission.wait_semaphores.iter() {
                signal_semaphores.push(vk::SemaphoreSubmitInfo {
                    semaphore: signal.semaphore.semaphore,
                    value: signal.value,
                    stage_mask: signal.stage_mask,
                    ..Default::default()
                });
            }
            for exec in submission.executables.iter() {
                command_buffers.push(vk::CommandBufferSubmitInfo {
                    command_buffer: exec.command_buffer.buffer,
                    ..Default::default()
                })
            }
        }

        let fence = Fence::new(self.queue.device.clone(), false)?;
        unsafe {
            self.queue
                .submit_raw2(submit_infos.as_slice(), fence.fence)?;
        }
        Ok(Some(QueueSubmissionFence {
            fence,
            _submissions: submissions,
        }))
    }
}

pub struct QueueSubmissionFence {
    fence: Fence,
    _submissions: Vec<Submission>,
}

impl QueueSubmissionFence {
    pub fn wait(self) -> VkResult<()> {
        self.fence.wait()?;
        drop(self);
        Ok(())
    }
}

pub struct SemaphoreOp {
    pub semaphore: Arc<Semaphore>,
    pub stage_mask: vk::PipelineStageFlags2,
    pub value: u64,
}
pub struct Submission {
    pub wait_semaphores: Vec<SemaphoreOp>,
    pub executables: Vec<CommandExecutable>,
    pub signal_semaphore: Vec<SemaphoreOp>,
}
