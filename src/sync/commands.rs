use std::sync::Arc;

use crate::{
    command::recorder::{CommandBufferBuilder, CommandExecutable, CommandRecorder},
    frames::{AcquiredFrame, FrameManager},
    queue::{
        semaphore::{Semaphore, TimelineSemaphore},
        QueueIndex, Queues, SemaphoreOp, StagedSemaphoreOp,
    },
};
use ash::{prelude::VkResult, vk};

use super::{semaphore::SemaphoreFuture, GPUFuture};

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
pub struct CommandsJoinFuture<'q, 'a> {
    // The future, and which pipeline stage that this future is going to block
    futures: Vec<(CommandsStageFuture<'q, 'a>, vk::PipelineStageFlags2)>,
}

impl<'q, 'a> CommandsStageFuture<'q, 'a> {
    pub fn then<T: GPUFuture>(mut self, mut next: T) -> T::NextFuture {
        if let Some(existing) = self.get_one_signaled_semaphore() {
            next.wait_semaphore(existing);
            return next.next_future();
        }
        let semaphore = self.pop_semaphore_pool();
        self.signal_semaphore(semaphore.clone());
        next.wait_semaphore(semaphore.clone());
        next.push_semaphore_pool(semaphore.increment());
        next.next_future()
    }
    /// self is going to block the execution of `block_stages` in the resulting future.
    /// other is going to block the execution of `other_stages` in the resulting future.
    pub fn join_commands(
        self,
        block_stages: vk::PipelineStageFlags2,
        other: CommandsStageFuture<'q, 'a>,
        other_stages: vk::PipelineStageFlags2,
    ) -> CommandsJoinFuture<'q, 'a> {
        // Add signals here
        CommandsJoinFuture {
            futures: vec![(self, block_stages), (other, other_stages)],
        }
    }
}

impl<'q, 'a> CommandsJoinFuture<'q, 'a> {
    // Still, the problem here is we still want to specify stages for each.
    pub fn then<'b>(self, mut next: CommandsFuture<'q>) -> CommandsFuture<'q> {
        for (mut fut, block_stages) in self.futures.into_iter() {
            if let Some(existing) = fut.get_one_signaled_semaphore() {
                next.stage(block_stages).wait_semaphore(existing);
                continue;
            }

            let semaphore = fut.pop_semaphore_pool();
            fut.signal_semaphore(semaphore.clone());
            next.stage(block_stages).wait_semaphore(semaphore.clone());
            next.push_semaphore_pool(semaphore.increment());
        }
        next
    }
    pub fn join_commands(
        mut self,
        next: CommandsStageFuture<'q, 'a>,
        block_stages: vk::PipelineStageFlags2,
    ) -> CommandsJoinFuture<'q, 'a> {
        // Add signals here
        self.futures.push((next, block_stages));
        self
    }
}

// OK, this is a perfect model.
// How would binary semaphores work?
// How would sparse binding work?
// How would generics work?

#[cfg(test)]
mod tests {
    use super::*;
    fn q() -> Queues {
        todo!()
    }
    #[test]
    fn test() {
        let queues = q();
        let mut task1 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task2 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task3 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);
        let mut task4 = CommandsFuture::new(&queues, QueueIndex(0), vec![]);

        task1
            .stage(vk::PipelineStageFlags2::VERTEX_SHADER)
            .then(task2.stage(vk::PipelineStageFlags2::TRANSFER));
        task1
            .stage(vk::PipelineStageFlags2::VERTEX_SHADER)
            .then(task3.stage(vk::PipelineStageFlags2::COMPUTE_SHADER))
            .stage(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .then(task4.stage(vk::PipelineStageFlags2::TRANSFER));
    }
}
