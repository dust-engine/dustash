use std::any::Any;
use std::collections::BinaryHeap;
use std::rc::Weak;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};
use ash::vk;

use crate::command::recorder::CommandRecorder;
use crate::command::sync::{AccessType, ImageBarrier, PipelineBarrier, MemoryBarrier};

use self::pipline_stage_order::access_is_write;
pub struct RenderGraph {
    heads: BinaryHeap<BinaryHeapKeyedEntry<Rc<RefCell<RenderGraphNode>>>>,
    resources: Vec<Box<dyn Send + Sync>>,
}
#[derive(Clone)]
struct BinaryHeapKeyedEntry<T>(isize, T);
impl<T> Ord for BinaryHeapKeyedEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl<T> PartialOrd for BinaryHeapKeyedEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> PartialEq for BinaryHeapKeyedEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl<T> Eq for BinaryHeapKeyedEntry<T> {
}

struct RenderGraphNode {
    nexts: Vec<BinaryHeapKeyedEntry<Rc<RefCell<RenderGraphNode>>>>,
    config: NodeConfig,
}

pub struct Then {
    heads: Vec<Weak<RefCell<RenderGraphNode>>>,
}

pub struct ResourceHandle<T> {
    idx: usize,
    _marker: PhantomData<T>
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            heads: BinaryHeap::new(),
            resources: Vec::new(),
        }
    }
    pub fn start<F: FnOnce(&mut NodeConfig) + 'static>(
        &mut self,
        run: F,
    ) -> Then {
        let mut config = NodeConfig::new();
        run(&mut config);
        let node = BinaryHeapKeyedEntry(config.priority, Rc::new(RefCell::new(RenderGraphNode {
            nexts: Vec::new(),
            config
        })));
        let head = Rc::downgrade(&node.1);
        self.heads.push(node);
        Then {
            heads: vec![head],
        }
    }

    pub fn import<T: Send + Sync + 'static>(&mut self, resource: T) -> ResourceHandle<T> {
        let idx = self.resources.len();
        self.resources.push(Box::new(resource));
        ResourceHandle { idx, _marker: PhantomData }
    }

    pub fn run(mut self) {
        let mut resources = self.resources.into_iter().map(|res| ResourceState {
            resource: res,
            dirty_stages: vk::PipelineStageFlags2::empty(),
            available_stages: vk::PipelineStageFlags2::empty()
        }).collect::<Vec<_>>();
        // TODO: finer grained optimizations when we have more complicated render graphs to test.
        while let Some(head) = self.heads.pop() {
            let mut head = match Rc::try_unwrap(head.1) {
                Ok(head) => head.into_inner(),
                Err(_) => panic!("Requries exclusive ownership"),
            };
            if let Some(record) = head.config.record {
                let mut ctx = RenderGraphContext::new();

                // For all reads and writes, if the image was dirty, flush and make not dirty.
                let mut dst_access: Vec<AccessType> = Vec::with_capacity(head.config.accesses.len());

                let mut global_barries: Vec<vk::MemoryBarrier2> = Vec::new();
                let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
                let mut buffer_barries: Vec<vk::BufferMemoryBarrier2> = Vec::new();

                for access in head.config.accesses.iter() {
                    let res = &mut resources[access.idx];
                    if !res.available_stages.contains(access.stage) {
                        // Needs to emit pipeline barrier.
                        match access.barrier {
                            Barrier::Image { src_layout, dst_layout, subresource_range } => image_barriers.push(vk::ImageMemoryBarrier2 {
                                src_stage_mask: res.dirty_stages,
                                src_access_mask: res.accesses,
                                dst_stage_mask: access.stage,
                                dst_access_mask: access.access,
                                old_layout: src_layout,
                                new_layout: dst_layout,
                                subresource_range,
                                image: todo!(), // Coerse res into image
                                ..Default::default()
                            }),
                            Barrier::Global => global_barries.push(vk::MemoryBarrier2 {
                                src_stage_mask: res.dirty_stages,
                                src_access_mask: res.accesses,
                                dst_stage_mask: access.stage,
                                dst_access_mask: access.access,
                                ..Default::default()
                            }),
                            Barrier::Buffer { subresource_range, offset, size } => buffer_barries.push(vk::BufferMemoryBarrier2 {
                                src_stage_mask: res.dirty_stages,
                                src_access_mask: res.accesses,
                                dst_stage_mask: access.stage,
                                dst_access_mask: access.access,
                                offset,
                                size,
                                buffer: todo!(),
                                ..Default::default()
                            }),
                        }
                        res.available_stages |= pipline_stage_order::logically_later_stages(access.stage); // and all subsequent stages
                        // RoR: the previous node should have the corresponding flags set in available_stages which prevents unnecessary pipeline barriers.
                        // WoW: After the first write, available stages is 0. Therefore, the pipeline barrier will be emitted.
                        // WoR: same as above.
                        // RoW. Only execution barrier required. Needs more consideration hre tomorrow.
                        // Also need to consider the case where you have multiple reads and writes in the same stage.
                    }

                    let is_write = access_is_write(access.access);
                    if is_write {
                        res.available_stages = vk::PipelineStageFlags2::empty();
                        if !res.prev_write {
                            res.accesses = vk::AccessFlags2::empty();
                            res.dirty_stages = vk::PipelineStageFlags2::empty();
                        }
                        res.accesses |= access.access;
                        res.dirty_stages |= access.stage;
                    }
                    res.prev_write = is_write;
                }
                CommandRecorder::simple_pipeline_barrier(todo!(), &PipelineBarrier::new(
                    memory_barrier, &[], &[], vk::DependencyFlags::empty()));

                // Execute the command.
                (record)(&mut ctx);
            }

            let new_heads = head.nexts.drain_filter(|next| Rc::strong_count(&mut next.1) == 1);
            self.heads.extend(new_heads);
        }
    }
}

pub struct ResourceState {
    resource: Box<dyn Send + Sync>,

    // If a stage writes to the resource, the corresponding bits will be set to True.
    dirty_stages: vk::PipelineStageFlags2,

    accesses: vk::AccessFlags2,
    
    // After a pipeline barrier, all dstStages gets set to true, indicating that the change is now visible to these stages.
    available_stages: vk::PipelineStageFlags2,

    prev_write: bool,
}

impl Then {
    pub fn then<F: FnOnce(&mut NodeConfig) + 'static>(
        &self,
        run: F,
    ) -> Then {
        let mut config = NodeConfig::new();
        run(&mut config);
        let node = BinaryHeapKeyedEntry(config.priority, Rc::new(RefCell::new(RenderGraphNode {
            nexts: Vec::new(),
            config
        })));
        let head = Rc::downgrade(&node.1);
        for head in self.heads.iter() {
            head
            .upgrade()
            .unwrap() // This should still be referenced by the graph itself.
            .borrow_mut()
            .nexts
            .push(node.clone());
        }
        Then {
            heads: vec![head],
        }
    }
    pub fn join(&self, other: &Then) -> Then {
        let mut heads = Vec::with_capacity(self.heads.len() + other.heads.len());
        heads.extend_from_slice(&self.heads);
        heads.extend_from_slice(&other.heads);
        Then {
            heads,
        }
    }
}


pub struct RenderGraphContext {

}
pub struct NodeConfig {
    /// The priority of the task.
    ///
    /// The value of this priority could be fine tuned to ensure maximum overlapping.
    /// However, a simple heuristic applies  here:
    /// A node should have lower priority if it consumes many results that are expensive to produce.
    /// A node should have higher priority if it is expensive to execute and its products are consumed by many dependent nodes.
    priority: isize,
    record: Option<Box<dyn FnOnce(&mut RenderGraphContext)>>,
    accesses: Vec<Access>,
}
impl NodeConfig {
    pub fn record(&mut self, record: impl FnOnce(&mut RenderGraphContext) + 'static) {
        self.record = Some(Box::new(record))
    }
    /// An image memory barrier
    pub fn image_access(
        &mut self,
        image: (),
        access: AccessType,
        src_layout: vk::ImageLayout,    // The image would be guaranteed to be in this layout before the node executes
        dst_layout: vk::ImageLayout,    // The image will be transitioned to this layout by this node
        subresource_range: vk::ImageSubresourceRange
    ) {
    }
    /// A global memory barrier
    pub fn access(
        &mut self,
        resource: (),
        access: AccessType,
    ) {

    }

    /// Note that BufferMemoryBarriers don't actually do anything useful beyond global memory barrier.
    /// That's why we always emit global memory barrier for buffer resources, and don't ask for buffer offset and size.
    pub fn buffer_access(
        &mut self,
        resource: (),
        access: AccessType,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {

    }
}
impl NodeConfig {
    fn new() -> Self {
        Self {
            priority: 0,
            record: None,
            accesses: Vec::new()
        }
    }
}
impl RenderGraphContext {
    fn new() -> Self {
        Self {}
    }
}

struct Access {
    idx: usize,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    barrier: Barrier
}
enum Barrier {
    Image {
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange
    },
    Global,
    Buffer {
        subresource_range: vk::ImageSubresourceRange,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        let mut graph = RenderGraph::new();
        let temp1 = graph
            .start(|ctx| {
                ctx.record(|ctx| {
                    println!("A");
                });
            })
            .then(|ctx| {
                ctx.record(|ctx| {
                    println!("B");
                });
            });

        let temp2 = temp1
            .then(|ctx| {
                ctx.record(|ctx| {
                    println!("C");
                });
            })
            .then(|ctx| {
                ctx.priority = 10000;
                ctx.record(|ctx| {
                    println!("D");
                });
            });
        let temp3 = temp1
            .then(|ctx| {
                ctx.priority = 100;
                ctx.record(|ctx| {
                    println!("E");
                });
            });

        let temp3 = temp1.join(&temp2).join(&temp3)
        .then(|ctx| {
            ctx.record(|ctx| {
                println!("Success");
            });
        });

        graph.run();
    }
}

mod pipline_stage_order {
    use ash::vk;
    use vk::PipelineStageFlags2 as F;

    const FRAGMENT_BITS: vk::PipelineStageFlags2 = F::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR | F::EARLY_FRAGMENT_TESTS | F::FRAGMENT_SHADER | F::LATE_FRAGMENT_TESTS | F::COLOR_ATTACHMENT_OUTPUT;
    const GRAPHICS_BITS: vk::PipelineStageFlags2 = F::DRAW_INDIRECT | F::INDEX_INPUT | F::VERTEX_ATTRIBUTE_INPUT |
    F::VERTEX_SHADER | F::TESSELLATION_CONTROL_SHADER | F::TESSELLATION_EVALUATION_SHADER | F::GEOMETRY_SHADER |
    F::TRANSFORM_FEEDBACK_EXT | FRAGMENT_BITS;

    const GRAPHICS_MESH_BITS: vk::PipelineStageFlags2 = F::DRAW_INDIRECT | F::TASK_SHADER_NV | F::MESH_SHADER_NV | FRAGMENT_BITS;

    const COMPUTE_BITS: vk::PipelineStageFlags2 = F::DRAW_INDIRECT | F::COMPUTE_SHADER;

    const RTX_BITS: vk::PipelineStageFlags2 = F::DRAW_INDIRECT | F::RAY_TRACING_SHADER_KHR;
    /// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-order
    pub(super) fn logically_later_stages(stage: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
        match stage {
            F::DRAW_INDIRECT => GRAPHICS_BITS | GRAPHICS_MESH_BITS | COMPUTE_BITS | RTX_BITS,
            F::INDEX_INPUT => GRAPHICS_BITS & !F::DRAW_INDIRECT,
            F::VERTEX_ATTRIBUTE_INPUT => GRAPHICS_BITS & !(F::DRAW_INDIRECT | F::INDEX_INPUT),
            F::VERTEX_SHADER => GRAPHICS_BITS & !(F::DRAW_INDIRECT | F::INDEX_INPUT | F::VERTEX_ATTRIBUTE_INPUT),
            F::TESSELLATION_CONTROL_SHADER => F::TESSELLATION_CONTROL_SHADER | F::TESSELLATION_EVALUATION_SHADER | F::GEOMETRY_SHADER |
            F::TRANSFORM_FEEDBACK_EXT | FRAGMENT_BITS,
            
            F::TESSELLATION_EVALUATION_SHADER => F::TESSELLATION_EVALUATION_SHADER | F::GEOMETRY_SHADER |
            F::TRANSFORM_FEEDBACK_EXT | FRAGMENT_BITS,
            F::GEOMETRY_SHADER => F::GEOMETRY_SHADER | F::TRANSFORM_FEEDBACK_EXT | FRAGMENT_BITS,
            F::TRANSFORM_FEEDBACK_EXT => F::TRANSFORM_FEEDBACK_EXT | FRAGMENT_BITS,
            F::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR => FRAGMENT_BITS,
            F::EARLY_FRAGMENT_TESTS => FRAGMENT_BITS & !(F::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR),
            F::FRAGMENT_SHADER => F::FRAGMENT_SHADER | F::LATE_FRAGMENT_TESTS | F::COLOR_ATTACHMENT_OUTPUT,
            F::LATE_FRAGMENT_TESTS => F::LATE_FRAGMENT_TESTS | F::COLOR_ATTACHMENT_OUTPUT,
            F::COLOR_ATTACHMENT_OUTPUT => F::COLOR_ATTACHMENT_OUTPUT,

            F::TASK_SHADER_NV => F::TASK_SHADER_NV | F::MESH_SHADER_NV | FRAGMENT_BITS,
            F::MESH_SHADER_NV => F::MESH_SHADER_NV | FRAGMENT_BITS,

            F::FRAGMENT_DENSITY_PROCESS_EXT => F::FRAGMENT_DENSITY_PROCESS_EXT | F::EARLY_FRAGMENT_TESTS,

            _ => stage,
        }
    }

    pub(super) fn access_is_write(access: vk::AccessFlags2) -> bool {
        use vk::AccessFlags2 as F;
        match access {
            F::SHADER_WRITE | F::COLOR_ATTACHMENT_WRITE | F::DEPTH_STENCIL_ATTACHMENT_WRITE |
            F::TRANSFER_WRITE | F::HOST_WRITE | F::MEMORY_WRITE | F::SHADER_STORAGE_WRITE => true,
            _ => false,
        }
    }
}
