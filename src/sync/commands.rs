use std::sync::Arc;

use crate::{
    command::recorder::CommandExecutable,
    queue::{semaphore::TimelineSemaphore, QueueIndex, Queues, SemaphoreOp, StagedSemaphoreOp},
};
use ash::vk;

use super::GPUFuture;

// This is not a GPU Future because it doesn't represent a momemnt. It's a pipeline.
pub struct CommandsFuture<'q> {
    queues: &'q Queues,
    pub(crate) queue: QueueIndex,
    pub(crate) available_semaphore_pool: Vec<SemaphoreOp>,
    pub(crate) semaphore_waits: Vec<StagedSemaphoreOp>,
    pub(crate) cmd_execs: Vec<Arc<CommandExecutable>>,
    pub(crate) semaphore_signals: Vec<StagedSemaphoreOp>,
}
impl Drop for CommandsFuture<'_> {
    fn drop(&mut self) {
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
impl<'q> CommandsFuture<'q> {
    fn pop_semaphore_pool(&mut self) -> SemaphoreOp {
        self.available_semaphore_pool.pop().unwrap_or_else(|| {
            let semaphore =
                TimelineSemaphore::new(self.queues.of_index(self.queue).device().clone(), 0)
                    .unwrap();
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
}
pub struct CommandsStageFuture<'q, 'a> {
    commands_future: &'a mut CommandsFuture<'q>,
    stage: vk::PipelineStageFlags2,
}

impl<'q, 'a> GPUFuture for CommandsStageFuture<'q, 'a> {
    fn pop_semaphore_pool(&mut self) -> SemaphoreOp {
        self.commands_future.pop_semaphore_pool()
    }
    fn push_semaphore_pool(&mut self, semaphore: SemaphoreOp) {
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
    fn get_one_signaled_semaphore(&self) -> Option<SemaphoreOp> {
        self.commands_future
            .semaphore_signals
            .iter()
            .find(|&s| s.is_timeline() && s.stage_mask == self.stage)
            .map(|s| s.clone().stageless())
    }

    type NextFuture = &'a mut CommandsFuture<'q>;

    fn next_future(self) -> Self::NextFuture {
        self.commands_future
    }
}

impl<'q> CommandsFuture<'q> {
    pub fn new(queues: &'q Queues, queue: QueueIndex, execs: Vec<Arc<CommandExecutable>>) -> Self {
        Self {
            queues,
            semaphore_signals: Vec::new(),
            semaphore_waits: Vec::new(),
            queue,
            cmd_execs: execs,
            available_semaphore_pool: Vec::new(),
        }
    }
    pub fn stage<'a>(&'a mut self, stage: vk::PipelineStageFlags2) -> CommandsStageFuture<'q, 'a> {
        CommandsStageFuture {
            commands_future: self,
            stage,
        }
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
        let queues = q();
        let mut task1 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task2 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task3 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task4 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);

        task1
            .stage(vk::PipelineStageFlags2::VERTEX_SHADER)
            .then(task2.stage(vk::PipelineStageFlags2::FRAGMENT_SHADER));
        task1
            .stage(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .then(task3.stage(vk::PipelineStageFlags2::FRAGMENT_SHADER))
            .stage(vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT)
            .then(task4.stage(vk::PipelineStageFlags2::COMPUTE_SHADER));
    }

    #[test]
    fn test_commands_to_commands_join() {
        let queues = q();
        let mut task1 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task2 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task3 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task4 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);

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
