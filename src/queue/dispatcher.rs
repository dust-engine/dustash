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
    commands: crossbeam::queue::SegQueue<QueueCommand>,
    command_count: AtomicUsize,
}

impl QueueDispatcher {
    pub fn new(queue: Queue, assigned_type: Option<QueueType>) -> Self {
        QueueDispatcher {
            queue,
            assigned_queue_type: assigned_type,
            commands: crossbeam::queue::SegQueue::new(),
            command_count: AtomicUsize::new(0),
        }
    }
    pub fn family_index(&self) -> u32 {
        self.queue.family_index
    }
    pub fn submit(
        &self,
        wait_semaphores: Vec<SemaphoreOp>,
        executables: Vec<CommandExecutable>,
        signal_semaphore: Vec<SemaphoreOp>,
    ) -> &Self {
        self.command_count.fetch_add(1, Ordering::Relaxed);
        self.commands.push(QueueCommand::Submit(Submission {
            wait_semaphores,
            executables,
            signal_semaphore,
        }));
        self
    }
    pub fn fence(&self, fence: Arc<Fence>) -> &Self {
        self.command_count.fetch_add(1, Ordering::Relaxed);
        self.commands.push(QueueCommand::Fence(fence));
        self
    }
    pub fn is_empty(&self) -> bool {
        self.command_count.load(Ordering::Relaxed) == 0
    }

    // TODO: bind_sparse

    // The returned QueueSubmissionFence needs to be waited on and dropped
    pub fn flush(&mut self) -> VkResult<()> {
        let num_submissions = *self.command_count.get_mut();
        if num_submissions == 0 {
            return Ok(());
        }
        *self.command_count.get_mut() = 0;
        let mut submissions: Vec<Submission> = Vec::new();
        let mut wait_semaphore_count: usize = 0;
        let mut signal_semaphore_count: usize = 0;
        let mut executables_count: usize = 0;
        {
            while let Some(op) = self.commands.pop() {
                match op {
                    QueueCommand::Submit(op) => {
                        wait_semaphore_count += op.wait_semaphores.len();
                        signal_semaphore_count += op.signal_semaphore.len();
                        executables_count += op.executables.len();
                        submissions.push(op);
                    }
                    QueueCommand::Fence(fence) => {
                        // Submit existing fences
                        let fenced_submissions = std::mem::replace(&mut submissions, Vec::new());
                        unsafe {
                            self.queue_submit(
                                fenced_submissions,
                                fence,
                                wait_semaphore_count,
                                signal_semaphore_count,
                                executables_count,
                            )?;
                        }
                        wait_semaphore_count = 0;
                        signal_semaphore_count = 0;
                        executables_count = 0;
                    }
                }
            }
        }
        // If there are still some unfenced submissions
        if submissions.len() > 0 {
            let fence = Fence::new(self.queue.device.clone(), false)?;
            let fence = Arc::new(fence);
            unsafe {
                self.queue_submit(
                    submissions,
                    fence,
                    wait_semaphore_count,
                    signal_semaphore_count,
                    executables_count,
                )?;
            }
        }
        Ok(())
    }

    unsafe fn queue_submit(
        &mut self,
        submissions: Vec<Submission>,
        fence: Arc<Fence>,
        wait_semaphore_count: usize,
        signal_semaphore_count: usize,
        executables_count: usize,
    ) -> VkResult<()> {
        let mut wait_semaphores: Vec<vk::SemaphoreSubmitInfo> =
            Vec::with_capacity(wait_semaphore_count);
        let mut signal_semaphores: Vec<vk::SemaphoreSubmitInfo> =
            Vec::with_capacity(signal_semaphore_count);
        let mut command_buffers: Vec<vk::CommandBufferSubmitInfo> =
            Vec::with_capacity(executables_count);
        let mut submit_infos: Vec<vk::SubmitInfo2> = Vec::with_capacity(submissions.len());

        for submission in submissions.iter() {
            submit_infos.push(vk::SubmitInfo2 {
                wait_semaphore_info_count: submission.wait_semaphores.len() as u32,
                p_wait_semaphore_infos: wait_semaphores.as_ptr().add(wait_semaphores.len()),
                command_buffer_info_count: submission.executables.len() as u32,
                p_command_buffer_infos: command_buffers.as_ptr().add(command_buffers.len()),
                signal_semaphore_info_count: submission.signal_semaphore.len() as u32,
                p_signal_semaphore_infos: signal_semaphores.as_ptr().add(signal_semaphores.len()),
                ..Default::default()
            });
            for wait in submission.wait_semaphores.iter() {
                wait_semaphores.push(vk::SemaphoreSubmitInfo {
                    semaphore: wait.semaphore.semaphore,
                    value: wait.value,
                    stage_mask: wait.stage_mask,
                    ..Default::default()
                });
            }
            for signal in submission.signal_semaphore.iter() {
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

        self.queue
            .submit_raw2(submit_infos.as_slice(), fence.fence)?;

        let submission = QueueSubmissionFence {
            fence,
            _submissions: submissions,
        };
        submission.wait_detached();
        Ok(())
    }
}

pub struct QueueSubmissionFence {
    fence: Arc<Fence>,
    _submissions: Vec<Submission>,
}

impl QueueSubmissionFence {
    pub fn wait(self) -> VkResult<()> {
        self.fence.wait()?;
        drop(self);
        Ok(())
    }

    pub fn wait_detached(self) {
        let task = blocking::unblock(|| {
            self.wait().unwrap();
        });
        task.detach();
    }
}

/// stage_mask in wait semaphores: Block the execution of these stages until the semaphore was signaled.
/// Stages not specified in wait_stages can proceed before the semaphore signal operation.
///
/// stage_mask in signal semaphores: Block the semaphore signal operation on the completion of these stages.
/// The semaphore will be signaled even if other stages are still running.
pub struct SemaphoreOp {
    pub semaphore: Arc<Semaphore>,
    pub stage_mask: vk::PipelineStageFlags2,
    pub value: u64,
}
impl SemaphoreOp {
    pub fn binary(semaphore: Arc<Semaphore>, stage_mask: vk::PipelineStageFlags2) -> Self {
        SemaphoreOp {
            semaphore,
            stage_mask,
            value: 0,
        }
    }
    pub fn timeline(
        semaphore: Arc<Semaphore>,
        stage_mask: vk::PipelineStageFlags2,
        value: u64,
    ) -> Self {
        SemaphoreOp {
            semaphore,
            stage_mask,
            value,
        }
    }
}

struct Submission {
    wait_semaphores: Vec<SemaphoreOp>,
    executables: Vec<CommandExecutable>,
    signal_semaphore: Vec<SemaphoreOp>,
}
enum QueueCommand {
    Submit(Submission),
    Fence(Arc<Fence>),
    // TODO: sparse bind
}
