use std::sync::Arc;

use crate::{
    command::{
        recorder::{CommandBufferBuilder, CommandExecutable, CommandRecorder},
        sync::PipelineBarrier,
    },
    queue::{
        semaphore::{TimelineSemaphore, TimelineSemaphoreOp},
        QueueIndex, Queues, SemaphoreOp, StagedSemaphoreOp,
    },
};
use ash::vk;

use super::GPUFuture;
use crate::HasDevice;

// This is not a GPU Future because it doesn't represent a momemnt. It's a pipeline.
pub struct CommandsFuture {
    queues: Arc<Queues>,
    pub(crate) queue: QueueIndex,
    pub(crate) available_semaphore_pool: Vec<TimelineSemaphoreOp>,
    pub(crate) semaphore_waits: Vec<StagedSemaphoreOp>,
    pub(crate) cmd_execs: Vec<Arc<CommandExecutable>>,
    pub(crate) semaphore_signals: Vec<StagedSemaphoreOp>,

    recording_cmd_buf: Option<CommandBufferBuilder>,
}
impl Drop for CommandsFuture {
    fn drop(&mut self) {
        self.flush_recording_commands();
        if self.cmd_execs.is_empty() {
            if self.semaphore_signals.is_empty() {
                return;
            }

            if self.semaphore_waits.is_empty() {
                // Signal directly.
                for signal in self.semaphore_signals.drain(..) {
                    if signal.is_timeline() {
                        signal.stageless().as_timeline().signal().unwrap();
                    } else {
                        unimplemented!("Can't signal a binary semaphore here...")
                    }
                }
                return;
            }
            // have both signal and waits
        }
        use std::mem::take;
        // When dropping CommandsFuture it is no longer possible to add semaphores to it.
        // Therefore, this is in fact the best opportunity to flush.
        self.queues.of_index(self.queue).submit(
            take(&mut self.semaphore_waits).into_boxed_slice(),
            take(&mut self.cmd_execs).into_boxed_slice(),
            take(&mut self.semaphore_signals).into_boxed_slice(),
        );
    }
}
impl CommandsFuture {
    pub fn new(queues: Arc<Queues>, queue: QueueIndex) -> Self {
        Self {
            queues,
            semaphore_signals: Vec::new(),
            semaphore_waits: Vec::new(),
            queue,
            cmd_execs: Vec::new(),
            available_semaphore_pool: Vec::new(),
            recording_cmd_buf: None,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.cmd_execs.len() == 0 && self.recording_cmd_buf.is_none()
    }
    fn pop_semaphore_pool(&mut self) -> TimelineSemaphoreOp {
        self.available_semaphore_pool.pop().unwrap_or_else(|| {
            let semaphore =
                TimelineSemaphore::new(self.queues.of_index(self.queue).device().clone(), 0)
                    .unwrap();
            let semaphore = Arc::new(semaphore);
            TimelineSemaphoreOp {
                semaphore: semaphore,
                value: 1,
            }
        })
    }
    fn push_semaphore_pool(&mut self, semaphore: TimelineSemaphoreOp) {
        self.available_semaphore_pool.push(semaphore);
    }

    pub fn then_command_exec(&mut self, command_exec: Arc<CommandExecutable>) -> &mut Self {
        self.cmd_execs.push(command_exec);
        self
    }

    pub fn then_commands<R>(&mut self, f: impl FnOnce(CommandRecorder) -> R) -> R {
        let mut recording_buffer = self.recording_cmd_buf.take().unwrap_or_else(|| {
            let buf = self
                .queues
                .of_index(self.queue)
                .shared_command_pool()
                .allocate_one()
                .unwrap();
            buf.start(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .unwrap()
        });
        let ret = recording_buffer.record(f);
        self.recording_cmd_buf = Some(recording_buffer);
        ret
    }
    fn flush_recording_commands(&mut self) {
        if let Some(builder) = self.recording_cmd_buf.take() {
            let exec = builder.end().unwrap();
            self.cmd_execs.push(Arc::new(exec));
        }
    }

    pub fn stage<'a>(&'a mut self, stage: vk::PipelineStageFlags2) -> CommandsStageFuture<'a> {
        CommandsStageFuture {
            commands_future: self,
            stage,
        }
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queues.of_index(self.queue).family_index()
    }
}

pub struct CommandsStageFuture<'a> {
    commands_future: &'a mut CommandsFuture,
    stage: vk::PipelineStageFlags2,
}

impl<'q, 'a> GPUFuture for CommandsStageFuture<'a> {
    fn pop_semaphore_pool(&mut self) -> TimelineSemaphoreOp {
        self.commands_future.pop_semaphore_pool()
    }
    fn push_semaphore_pool(&mut self, semaphore: TimelineSemaphoreOp) {
        self.commands_future.push_semaphore_pool(semaphore)
    }
    fn wait_semaphore(&mut self, semaphore: SemaphoreOp) {
        self.commands_future
            .semaphore_waits
            .push(semaphore.staged(self.stage));
    }
    fn signal_semaphore(&mut self, semaphore: SemaphoreOp) {
        self.commands_future
            .semaphore_signals
            .push(semaphore.staged(self.stage));
    }

    /// Returns one signaled semaphore.
    fn get_one_signaled_semaphore(&self) -> Option<TimelineSemaphoreOp> {
        self.commands_future
            .semaphore_signals
            .iter()
            .find(|&s| s.is_timeline() && s.stage_mask == self.stage)
            .map(|s| s.clone().stageless().as_timeline())
    }
}

impl<'q, 'a> CommandsStageFuture<'a> {
    /// Specify a new queue for execution.
    /// When the new queue and the old queue has the same queue family, this does nothing.
    /// `stage`: This stage in following commands will be blocked by the
    /// pipeline barrier until the queue transfer is complete.
    pub fn then_queue_transfer(
        mut self,
        new_queue: QueueIndex,
        barrier: &PipelineBarrier,
        stage: vk::PipelineStageFlags2,
    ) -> &'a mut CommandsFuture {
        let old_index = self
            .commands_future
            .queues
            .of_index(self.commands_future.queue)
            .family_index();
        let new_index = self
            .commands_future
            .queues
            .of_index(new_queue)
            .family_index();
        if old_index == new_index {
            return self.commands_future;
        }
        self.commands_future.then_commands(|mut recorder| {
            recorder.simple_pipeline_barrier2(barrier);
        });
        let mut future = CommandsFuture::new(self.commands_future.queues.clone(), new_queue); // The future on the new queue
        self.then(future.stage(stage));
        std::mem::swap(&mut future, self.commands_future);
        // future is now the old future.
        drop(future);

        self.commands_future.then_commands(|mut recorder| {
            recorder.simple_pipeline_barrier2(barrier);
        });
        self.commands_future
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn q() -> Queues {
        todo!()
    }
    #[test]
    fn test_commands_to_commands_split() {
        let queues = Arc::new(q());
        let mut task1 = CommandsFuture::new(queues.clone(), QueueIndex(0));
        let mut task2 = CommandsFuture::new(queues.clone(), QueueIndex(0));
        let mut task3 = CommandsFuture::new(queues.clone(), QueueIndex(0));
        let mut task4 = CommandsFuture::new(queues.clone(), QueueIndex(0));

        task1
            .stage(vk::PipelineStageFlags2::VERTEX_SHADER)
            .then(task2.stage(vk::PipelineStageFlags2::FRAGMENT_SHADER));
        task1
            .stage(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .then(task3.stage(vk::PipelineStageFlags2::FRAGMENT_SHADER));
        task1
            .stage(vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT)
            .then(task4.stage(vk::PipelineStageFlags2::COMPUTE_SHADER));
    }

    #[test]
    fn test_commands_to_commands_join() {
        let queues = Arc::new(q());
        let mut task1 = CommandsFuture::new(queues.clone(), QueueIndex(0));
        let mut task2 = CommandsFuture::new(queues.clone(), QueueIndex(0));
        let mut task3 = CommandsFuture::new(queues.clone(), QueueIndex(0));
        let mut task4 = CommandsFuture::new(queues.clone(), QueueIndex(0));

        task1
            .stage(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .then(task3.stage(vk::PipelineStageFlags2::COMPUTE_SHADER));
        task2
            .stage(vk::PipelineStageFlags2::VERTEX_SHADER)
            .then(task3.stage(vk::PipelineStageFlags2::COMPUTE_SHADER));

        task3
            .stage(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .then(task4.stage(vk::PipelineStageFlags2::BLIT));
    }
}
