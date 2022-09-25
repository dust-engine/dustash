use std::collections::BinaryHeap;
use std::rc::Weak;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};
use ash::vk;

use crate::command::recorder::CommandRecorder;
use crate::resources::{HasBuffer, HasImage};

use self::pipline_stage_order::access_is_write;
pub struct RenderGraph {
    heads: BinaryHeap<BinaryHeapKeyedEntry<Rc<RefCell<RenderGraphNode>>>>,
    resources: Vec<Resource>,
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
        self.resources.push(Resource::Other(Box::new(resource)));
        ResourceHandle { idx, _marker: PhantomData }
    }

    pub fn import_image<T: HasImage + Send + Sync + 'static>(&mut self, resource: T) -> ResourceHandle<T> {
        let idx = self.resources.len();
        self.resources.push(Resource::Other(Box::new(resource)));
        ResourceHandle { idx, _marker: PhantomData }
    }

    pub fn import_buffer<T: HasBuffer + Send + Sync + 'static>(&mut self, resource: T) -> ResourceHandle<T> {
        let idx = self.resources.len();
        self.resources.push(Resource::Other(Box::new(resource)));
        ResourceHandle { idx, _marker: PhantomData }
    }

    pub fn run(mut self) {
        let mut ctx = RenderGraphContext::new();
        let mut resources = self.resources.into_iter().map(|res| ResourceState {
            resource: res,
            dirty_stages: vk::PipelineStageFlags2::empty(),
            accesses: vk::AccessFlags2::empty(),
            prev_write: false,
            available_stages: vk::PipelineStageFlags2::empty(),
            available_accesses: vk::AccessFlags2::empty(),
            layout: vk::ImageLayout::UNDEFINED
        }).collect::<Vec<_>>();
        while let Some(head) = self.heads.pop() {
            let mut head = match Rc::try_unwrap(head.1) {
                Ok(head) => head.into_inner(),
                Err(_) => panic!("Requries exclusive ownership"),
            };
            if let Some(record) = head.config.record {

                let mut global_barries: Vec<vk::MemoryBarrier2> = Vec::new();
                let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
                let mut buffer_barries: Vec<vk::BufferMemoryBarrier2> = Vec::new();

                for access in head.config.accesses.iter() {
                    let res = &mut resources[access.idx];
                    
                    let is_write = access_is_write(access.access);

                    let mut add_barrier = |src_access_mask: vk::AccessFlags2, dst_access_mask: vk::AccessFlags2| {
                        match access.barrier {
                            Barrier::Image { src_layout, dst_layout, subresource_range } => {
                                image_barriers.push(vk::ImageMemoryBarrier2 {
                                    src_stage_mask: res.dirty_stages,
                                    src_access_mask,
                                    dst_stage_mask: access.stage,
                                    dst_access_mask,
                                    old_layout: res.layout,
                                    new_layout: src_layout,
                                    subresource_range,
                                    image: match &res.resource {
                                        Resource::Image(image) => image.raw_image(),
                                        _ => panic!()
                                    },
                                    ..Default::default()
                                });
                                res.layout = dst_layout;
                            },
                            Barrier::Global => global_barries.push(vk::MemoryBarrier2 {
                                src_stage_mask: res.dirty_stages,
                                src_access_mask,
                                dst_stage_mask: access.stage,
                                dst_access_mask,
                                ..Default::default()
                            }),
                            Barrier::Buffer {offset, size } => buffer_barries.push(vk::BufferMemoryBarrier2 {
                                src_stage_mask: res.dirty_stages,
                                src_access_mask,
                                dst_stage_mask: access.stage,
                                dst_access_mask,
                                offset,
                                size,
                                buffer: match &res.resource {
                                    Resource::Buffer(image) => image.raw_buffer(),
                                    _ => panic!()
                                },
                                ..Default::default()
                            }),
                        }
                    };
                    match (res.prev_write, is_write) {
                        (true, true) => {
                            // Write after write.
                            // needs execution and memory barrier.
                            add_barrier(res.accesses, access.access);
                            res.dirty_stages = access.stage;
                            res.accesses = access.access;
                            res.available_stages = vk::PipelineStageFlags2::empty();
                            res.available_accesses = vk::AccessFlags2::empty();
                        },
                        (true, false) => {
                            // Read after write.
                            add_barrier(res.accesses, access.access);
                            res.available_stages = pipline_stage_order::logically_later_stages(access.stage);
                            res.available_accesses = access.access;
                        },
                        (false, true) => {
                            // Write after read
                            // Execution barrier only.
                            add_barrier(vk::AccessFlags2::empty(), vk::AccessFlags2::empty());
                            res.dirty_stages = access.stage;
                            res.accesses = access.access;
                            res.available_stages = vk::PipelineStageFlags2::empty();
                            res.available_accesses = vk::AccessFlags2::empty();
                        },
                        (false, false) => {
                            // Read after read. Only emit barrier when it's not already covered.
                            if !res.available_stages.contains(access.stage) || !res.available_accesses.contains(access.access) {
                                // Re-emit barrier.
                                add_barrier(res.accesses, access.access);
                                res.available_stages |= pipline_stage_order::logically_later_stages(access.stage);
                                res.available_accesses |= access.access;
                            }
                        },
                    }
                    res.prev_write = is_write;
                }
                unsafe {
                    ctx.command_recorder.pipeline_barrier2(
                        &vk::DependencyInfo {
                            dependency_flags: vk::DependencyFlags::BY_REGION,
                            memory_barrier_count: global_barries.len() as u32,
                            p_memory_barriers: global_barries.as_ptr(),
                            buffer_memory_barrier_count: buffer_barries.len() as u32,
                            p_buffer_memory_barriers: buffer_barries.as_ptr(),
                            image_memory_barrier_count: image_barriers.len() as u32,
                            p_image_memory_barriers: image_barriers.as_ptr(),
                            ..Default::default()
                        }
                    );
                }
                // Execute the command.
                (record)(&mut ctx);
            }

            let new_heads = head.nexts.drain_filter(|next| Rc::strong_count(&mut next.1) == 1);
            self.heads.extend(new_heads);
        }
        ctx.command_recorder.referenced_resources.extend(resources.into_iter().map(|a| match a.resource {
            Resource::Other(res) => res,
            Resource::Buffer(buffer) => buffer.boxed_type_erased(),
            Resource::Image(image) => image.boxed_type_erased(),
        }));
    }
}

pub enum Resource {
    Buffer(Box<dyn HasBuffer + Send + Sync + 'static>),
    Image(Box<dyn HasImage + Send + Sync + 'static>),
    Other(Box<dyn Send + Sync + 'static>),
}

pub struct ResourceState {
    resource: Resource,

    // If a stage writes to the resource, the corresponding bits will be set to True.
    dirty_stages: vk::PipelineStageFlags2,

    accesses: vk::AccessFlags2,
    
    // After a pipeline barrier, all dstStages gets set to true, indicating that the change is now visible to these stages.
    available_stages: vk::PipelineStageFlags2,
    available_accesses: vk::AccessFlags2,

    prev_write: bool,

    layout: vk::ImageLayout,
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


pub struct RenderGraphContext<'a> {
    command_recorder: CommandRecorder<'a>
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
    pub fn record(&mut self, record: impl FnOnce(&mut RenderGraphContext) + 'static) -> &mut Self {
        self.record = Some(Box::new(record));
        self
    }
    /// An image memory barrier
    pub fn image_access<T: HasImage>(
        &mut self,
        image: ResourceHandle<T>,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        src_layout: vk::ImageLayout,    // The image would be guaranteed to be in this layout before the node executes
        dst_layout: vk::ImageLayout,    // The image will be transitioned to this layout by this node
        subresource_range: vk::ImageSubresourceRange
    ) -> &mut Self {
        self.accesses.push(Access {
            idx: image.idx,
            stage,
            access,
            barrier: Barrier::Image { src_layout, dst_layout, subresource_range },
        });
        self
    }
    /// A global memory barrier
    pub fn access<T>(
        &mut self,
        resource: ResourceHandle<T>,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> &mut Self {
        self.accesses.push(Access {
            idx: resource.idx,
            stage,
            access,
            barrier: Barrier::Global,
        });
        self
    }

    /// Note that BufferMemoryBarriers don't actually do anything useful beyond global memory barrier.
    /// That's why we always emit global memory barrier for buffer resources, and don't ask for buffer offset and size.
    pub fn buffer_access<T: HasBuffer>(
        &mut self,
        resource: ResourceHandle<T>,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> &mut Self {
        self.accesses.push(Access {
            idx: resource.idx,
            stage,
            access,
            barrier: Barrier::Buffer { offset, size },
        });
        self
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
impl<'a> RenderGraphContext<'a> {
    fn new() -> Self {
        Self {
            command_recorder: todo!()
        }
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

    const FRAGMENT_BITS: vk::PipelineStageFlags2 = F::from_raw(F::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR.as_raw() | F::EARLY_FRAGMENT_TESTS.as_raw() | F::FRAGMENT_SHADER.as_raw() | F::LATE_FRAGMENT_TESTS.as_raw() | F::COLOR_ATTACHMENT_OUTPUT.as_raw());
    const GRAPHICS_BITS: vk::PipelineStageFlags2 = F::from_raw(F::DRAW_INDIRECT.as_raw() | F::INDEX_INPUT.as_raw() | F::VERTEX_ATTRIBUTE_INPUT.as_raw() |
    F::VERTEX_SHADER.as_raw() | F::TESSELLATION_CONTROL_SHADER.as_raw() | F::TESSELLATION_EVALUATION_SHADER.as_raw() | F::GEOMETRY_SHADER.as_raw() |
    F::TRANSFORM_FEEDBACK_EXT.as_raw() | FRAGMENT_BITS.as_raw());

    const GRAPHICS_MESH_BITS: vk::PipelineStageFlags2 = F::from_raw(F::DRAW_INDIRECT.as_raw() | F::TASK_SHADER_NV.as_raw() | F::MESH_SHADER_NV.as_raw() | FRAGMENT_BITS.as_raw());

    const COMPUTE_BITS: vk::PipelineStageFlags2 = F::from_raw(F::DRAW_INDIRECT.as_raw() | F::COMPUTE_SHADER.as_raw());

    const RTX_BITS: vk::PipelineStageFlags2 = F::from_raw(F::DRAW_INDIRECT.as_raw() | F::RAY_TRACING_SHADER_KHR.as_raw());
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
        access.intersects(F::SHADER_WRITE | F::COLOR_ATTACHMENT_WRITE | F::DEPTH_STENCIL_ATTACHMENT_WRITE |
            F::TRANSFER_WRITE | F::HOST_WRITE | F::MEMORY_WRITE | F::SHADER_STORAGE_WRITE)
    }
}
