use std::{
    fmt::Debug,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use ash::{prelude::VkResult, vk};

use crate::{command::recorder::CommandExecutable, fence::Fence, Device};

#[cfg(feature = "shared_command_pool")]
use crate::command::shared_pool::SharedCommandPool;

use super::{
    router::QueueIndex,
    semaphore::{Semaphore, TimelineSemaphoreOp},
    Queue, QueueType,
};

/// Queue operations require exclusive access to the Queue object, and it's usually more
/// performant to batch queue submissions together.
/// The QueueDispatcher provides shared references to a Queue, and acts like a buffer
/// so that the queue operations can be submitted in batch on a frame-to-frame basis.
pub struct QueueDispatcher {
    pub(crate) queue: Queue,
    index: QueueIndex,
    assigned_queue_type: Option<QueueType>,
    commands: crossbeam_queue::SegQueue<QueueCommand>,
    command_count: AtomicUsize,

    #[cfg(feature = "shared_command_pool")]
    shared_command_pool: SharedCommandPool,
}

impl crate::HasDevice for QueueDispatcher {
    fn device(&self) -> &Arc<Device> {
        self.queue.device()
    }
}

impl QueueDispatcher {
    pub fn new(queue: Queue, assigned_type: Option<QueueType>, index: QueueIndex) -> Self {
        QueueDispatcher {
            assigned_queue_type: assigned_type,
            commands: crossbeam_queue::SegQueue::new(),
            command_count: AtomicUsize::new(0),
            #[cfg(feature = "shared_command_pool")]
            shared_command_pool: SharedCommandPool::new(&queue),
            queue,
            index,
        }
    }
    pub fn shared_command_pool(&self) -> &SharedCommandPool {
        &self.shared_command_pool
    }
    pub fn family_index(&self) -> u32 {
        self.queue.family_index
    }
    pub fn index(&self) -> QueueIndex {
        self.index
    }
    pub fn submit(
        &self,
        wait_semaphores: Box<[StagedSemaphoreOp]>,
        executables: Box<[Arc<CommandExecutable>]>,
        signal_semaphores: Box<[StagedSemaphoreOp]>,
    ) -> &Self {
        self.command_count.fetch_add(1, Ordering::Relaxed);
        self.commands.push(QueueCommand::Submit(Submission {
            wait_semaphores,
            executables,
            signal_semaphores,
        }));
        self
    }
    pub fn is_empty(&self) -> bool {
        self.command_count.load(Ordering::Relaxed) == 0
    }

    pub fn sparse_bind(
        &self,
        wait_semaphores: Box<[SemaphoreOp]>,
        buffer_binds: Box<[(vk::Buffer, Box<[vk::SparseMemoryBind]>)]>,
        image_opaque_binds: Box<[(vk::Image, Box<[vk::SparseMemoryBind]>)]>,
        image_binds: Box<[(vk::Image, Box<[vk::SparseImageMemoryBind]>)]>,
        signal_semaphores: Box<[SemaphoreOp]>,
    ) -> &Self {
        self.command_count.fetch_add(1, Ordering::Relaxed);
        self.commands.push(QueueCommand::BindSparse(BindSparse {
            wait_semaphores,
            buffer_binds,
            image_binds,
            image_opaque_binds,
            signal_semaphores,
        }));
        self
    }

    #[must_use = "Call block(), wait() or wait_detached() to ensure that the render resources are released after submission completion."]
    pub fn flush(&mut self) -> VkResult<Option<QueueSubmissionFence>> {
        let num_submissions = *self.command_count.get_mut();
        if num_submissions == 0 {
            return Ok(None);
        }
        *self.command_count.get_mut() = 0;
        let mut submissions: Vec<Submission> = Vec::new();
        let mut wait_semaphore_count: usize = 0;
        let mut signal_semaphore_count: usize = 0;
        let mut executables_count: usize = 0;

        let mut bind_semaphore_count: usize = 0;
        let mut buffer_bind_count: usize = 0;
        let mut image_opaque_bind_count: usize = 0;
        let mut image_bind_count: usize = 0;
        let mut binds: Vec<BindSparse> = Vec::new();
        {
            while let Some(op) = self.commands.pop() {
                match op {
                    QueueCommand::Submit(op) => {
                        wait_semaphore_count += op.wait_semaphores.len();
                        signal_semaphore_count += op.signal_semaphores.len();
                        executables_count += op.executables.len();
                        submissions.push(op);
                    }
                    QueueCommand::BindSparse(bind) => {
                        bind_semaphore_count +=
                            bind.signal_semaphores.len() + bind.wait_semaphores.len();
                        image_bind_count += bind.image_binds.len();
                        image_opaque_bind_count += bind.image_opaque_binds.len();
                        buffer_bind_count += bind.buffer_binds.len();
                        binds.push(bind);
                    }
                }
            }
        }
        if !binds.is_empty() {
            unsafe {
                self.queue_bind_sparse(
                    binds,
                    bind_semaphore_count,
                    buffer_bind_count,
                    image_opaque_bind_count,
                    image_bind_count,
                )?;
            }
        }
        // If there are still some unfenced submissions
        assert!(!submissions.is_empty());
        let fence = Fence::new(self.queue.device.clone(), false)?;
        let task = unsafe {
            self.queue_submit(
                submissions,
                fence,
                wait_semaphore_count,
                signal_semaphore_count,
                executables_count,
            )?
        };
        Ok(Some(task))
    }

    unsafe fn queue_bind_sparse(
        &mut self,
        binds: Vec<BindSparse>,
        num_semaphores: usize,
        buffer_bind_count: usize,
        image_opaque_bind_count: usize,
        image_bind_count: usize,
    ) -> VkResult<()> {
        let mut semaphores: Vec<vk::Semaphore> = Vec::with_capacity(num_semaphores);
        let mut semaphore_values: Vec<u64> = Vec::with_capacity(num_semaphores);
        let mut infos: Vec<vk::BindSparseInfo> = Vec::with_capacity(binds.len());
        let mut timeline_infos: Vec<vk::TimelineSemaphoreSubmitInfo> =
            Vec::with_capacity(binds.len());

        let mut buffer_binds: Vec<vk::SparseBufferMemoryBindInfo> =
            Vec::with_capacity(buffer_bind_count);
        let mut image_opaque_binds: Vec<vk::SparseImageOpaqueMemoryBindInfo> =
            Vec::with_capacity(image_opaque_bind_count);
        let mut image_binds: Vec<vk::SparseImageMemoryBindInfo> =
            Vec::with_capacity(image_bind_count);

        // Collect bind infos into vecs while converting into vk formats.
        for info in binds.iter() {
            let p_wait_semaphores = semaphores.as_ptr().add(semaphores.len());
            semaphores.extend(info.wait_semaphores.iter().map(|w| w.semaphore.semaphore));
            let p_signal_semaphores = semaphores.as_ptr().add(semaphores.len());
            semaphores.extend(info.signal_semaphores.iter().map(|w| w.semaphore.semaphore));

            let p_wait_semaphore_values = semaphore_values.as_ptr().add(semaphore_values.len());
            semaphore_values.extend(info.wait_semaphores.iter().map(|w| w.value));
            let p_signal_semaphore_values = semaphore_values.as_ptr().add(semaphore_values.len());
            semaphore_values.extend(info.signal_semaphores.iter().map(|w| w.value));

            let p_timeline_info = timeline_infos.as_ptr().add(timeline_infos.len());
            timeline_infos.push(vk::TimelineSemaphoreSubmitInfo {
                wait_semaphore_value_count: info.wait_semaphores.len() as u32,
                p_wait_semaphore_values,
                signal_semaphore_value_count: info.signal_semaphores.len() as u32,
                p_signal_semaphore_values,
                ..Default::default()
            });

            let p_buffer_binds = buffer_binds.as_ptr().add(buffer_binds.len());
            buffer_binds.extend(info.buffer_binds.iter().map(|(buffer, bind)| {
                vk::SparseBufferMemoryBindInfo {
                    buffer: *buffer,
                    bind_count: bind.len() as u32,
                    p_binds: bind.as_ptr(),
                }
            }));
            let p_image_opaque_binds = image_opaque_binds.as_ptr().add(image_opaque_binds.len());
            image_opaque_binds.extend(info.image_opaque_binds.iter().map(|(image, bind)| {
                vk::SparseImageOpaqueMemoryBindInfo {
                    image: *image,
                    bind_count: bind.len() as u32,
                    p_binds: bind.as_ptr(),
                }
            }));
            let p_image_binds = image_binds.as_ptr().add(image_binds.len());
            image_binds.extend(info.image_binds.iter().map(|(image, bind)| {
                vk::SparseImageMemoryBindInfo {
                    image: *image,
                    bind_count: bind.len() as u32,
                    p_binds: bind.as_ptr(),
                }
            }));

            infos.push(vk::BindSparseInfo {
                wait_semaphore_count: info.wait_semaphores.len() as u32,
                p_wait_semaphores,
                buffer_bind_count: info.buffer_binds.len() as u32,
                p_buffer_binds,
                image_opaque_bind_count: info.image_opaque_binds.len() as u32,
                p_image_opaque_binds,
                image_bind_count: info.image_binds.len() as u32,
                p_image_binds,
                signal_semaphore_count: info.signal_semaphores.len() as u32,
                p_signal_semaphores,
                p_next: p_timeline_info as *const _,
                ..Default::default()
            })
        }
        self.queue
            .bind_sparse(infos.as_slice(), vk::Fence::null())?;
        Ok(())
    }

    unsafe fn queue_submit(
        &mut self,
        submissions: Vec<Submission>,
        fence: Fence,
        wait_semaphore_count: usize,
        signal_semaphore_count: usize,
        executables_count: usize,
    ) -> VkResult<QueueSubmissionFence> {
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
                signal_semaphore_info_count: submission.signal_semaphores.len() as u32,
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
            for signal in submission.signal_semaphores.iter() {
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
            fences: vec![fence],
            submissions: submissions,
        };
        Ok(submission)
    }
}

pub struct QueueSubmissionFence {
    fences: Vec<Fence>,
    submissions: Vec<Submission>,
}

impl QueueSubmissionFence {
    pub fn new() -> Self {
        Self {
            fences: Vec::new(),
            submissions: Vec::new(),
        }
    }
    pub fn block(self) -> VkResult<()> {
        Fence::join(self.fences).wait().unwrap();
        drop(self.submissions);
        Ok(())
    }
    pub fn wait(self) -> blocking::Task<()> {
        blocking::unblock(|| {
            self.block().unwrap();
        })
    }

    pub fn wait_detached(self) {
        let task = self.wait();
        task.detach();
    }
    pub fn merge(&mut self, mut other: Self) {
        self.fences.append(&mut other.fences);
        self.submissions.append(&mut other.submissions);
    }
}

/// stage_mask in wait semaphores: Block the execution of these stages until the semaphore was signaled.
/// Stages not specified in wait_stages can proceed before the semaphore signal operation.
///
/// stage_mask in signal semaphores: Block the semaphore signal operation on the completion of these stages.
/// The semaphore will be signaled even if other stages are still running.
#[derive(Clone)]
pub struct StagedSemaphoreOp {
    pub semaphore: Arc<Semaphore>,
    pub stage_mask: vk::PipelineStageFlags2,
    // When value == 0, the `StagedSemaphoreOp` is a binary semaphore.
    pub value: u64,
}
impl Debug for StagedSemaphoreOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = if self.is_timeline() {
            "StagedSemaphoreOp(Timeline)"
        } else {
            "StagedSemaphoreOp(Binary)"
        };
        let mut res = f.debug_tuple(name);
        res.field(&self.semaphore.semaphore).field(&self.stage_mask);
        if self.is_timeline() {
            res.field(&self.value);
        }
        res.finish()
    }
}

#[derive(Clone)]
pub struct SemaphoreOp {
    pub semaphore: Arc<Semaphore>,
    pub value: u64,
}

impl SemaphoreOp {
    pub fn staged(self, stage: vk::PipelineStageFlags2) -> StagedSemaphoreOp {
        StagedSemaphoreOp {
            semaphore: self.semaphore,
            stage_mask: stage,
            value: self.value,
        }
    }
    pub fn increment(self) -> Self {
        Self {
            semaphore: self.semaphore,
            value: self.value + 1,
        }
    }
    pub fn is_timeline(&self) -> bool {
        // Because the semaphore value is always >= 0, signaling a semaphore to be 0
        // or waiting a semaphore to turn 0 is meaningless.
        self.value != 0
    }
    pub fn as_timeline(self) -> TimelineSemaphoreOp {
        assert!(self.is_timeline());
        TimelineSemaphoreOp {
            semaphore: unsafe { self.semaphore.as_timeline_arc() },
            value: self.value,
        }
    }
}

impl StagedSemaphoreOp {
    pub fn binary(semaphore: Arc<Semaphore>, stage_mask: vk::PipelineStageFlags2) -> Self {
        StagedSemaphoreOp {
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
        StagedSemaphoreOp {
            semaphore,
            stage_mask,
            value,
        }
    }
    pub fn is_timeline(&self) -> bool {
        // Because the semaphore value is always >= 0, signaling a semaphore to be 0
        // or waiting a semaphore to turn 0 is meaningless.
        self.value != 0
    }
    pub fn stageless(self) -> SemaphoreOp {
        SemaphoreOp {
            semaphore: self.semaphore,
            value: self.value,
        }
    }
    pub fn increment(self) -> Self {
        Self {
            semaphore: self.semaphore,
            value: self.value + 1,
            stage_mask: self.stage_mask,
        }
    }
}

#[derive(Debug)]
struct Submission {
    wait_semaphores: Box<[StagedSemaphoreOp]>,
    executables: Box<[Arc<CommandExecutable>]>,
    signal_semaphores: Box<[StagedSemaphoreOp]>,
}
struct BindSparse {
    wait_semaphores: Box<[SemaphoreOp]>,
    buffer_binds: Box<[(vk::Buffer, Box<[vk::SparseMemoryBind]>)]>,
    image_opaque_binds: Box<[(vk::Image, Box<[vk::SparseMemoryBind]>)]>,
    image_binds: Box<[(vk::Image, Box<[vk::SparseImageMemoryBind]>)]>,
    signal_semaphores: Box<[SemaphoreOp]>,
}
enum QueueCommand {
    Submit(Submission),
    BindSparse(BindSparse),
}
