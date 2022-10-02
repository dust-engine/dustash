use crate::command::recorder::CommandRecorder;

pub mod cache;
pub mod pipeline;
pub mod sbt;

impl<'a> CommandRecorder<'a> {
    pub fn trace_rays(&mut self, sbt: &sbt::Sbt, width: u32, height: u32, depth: u32) {
        unsafe {
            sbt.pipeline.loader.cmd_trace_rays(
                self.command_buffer,
                &sbt.raygen_sbt,
                &sbt.miss_sbt,
                &sbt.hit_sbt,
                &sbt.callable_sbt,
                width,
                height,
                depth,
            )
        }
    }
}
