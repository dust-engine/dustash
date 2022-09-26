use std::sync::Arc;

use crate::ray_tracing::pipeline::RayTracingPipeline;

use super::RenderGraphContext;

pub trait RenderPass {}

struct RayTracingRenderPass {
    pipeline: Arc<RayTracingPipeline>,
}

impl RayTracingRenderPass {
    pub fn run(ctx: &mut RenderGraphContext) {}
}
