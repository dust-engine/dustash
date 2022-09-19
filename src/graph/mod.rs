use std::any::Any;
use std::collections::BinaryHeap;
use std::rc::Weak;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};
use ash::vk;
pub struct RenderGraph {
    heads: BinaryHeap<BinaryHeapKeyedEntry<Rc<RefCell<RenderGraphNode>>>>,
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
    record: Option<Box<dyn FnOnce(&mut RenderGraphContext)>>,
}

pub struct Then {
    heads: Vec<Weak<RefCell<RenderGraphNode>>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self { heads: BinaryHeap::new() }
    }
    pub fn start<F: FnOnce(&mut NodeConfig) + 'static>(
        &mut self,
        run: F,
    ) -> Then {
        let mut config = NodeConfig::new();
        run(&mut config);
        let node = BinaryHeapKeyedEntry(config.priority, Rc::new(RefCell::new(RenderGraphNode {
            nexts: Vec::new(),
            record: config.record,
        })));
        let head = Rc::downgrade(&node.1);
        self.heads.push(node);
        Then {
            heads: vec![head],
        }
    }

    pub fn run(mut self) {
        while let Some(head) = self.heads.pop() {
            let mut head = match Rc::try_unwrap(head.1) {
                Ok(head) => head.into_inner(),
                Err(_) => panic!("Requries exclusive ownership"),
            };
            if let Some(record) = head.record {
                let mut ctx = RenderGraphContext::new();
                (record)(&mut ctx);
            }

            let new_heads = head.nexts.drain_filter(|next| Rc::strong_count(&mut next.1) == 1);
            self.heads.extend(new_heads);
        }
    }
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
            record: config.record,
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
    record: Option<Box<dyn FnOnce(&mut RenderGraphContext)>>
}
impl NodeConfig {
    pub fn record(&mut self, record: impl FnOnce(&mut RenderGraphContext) + 'static) {
        self.record = Some(Box::new(record))
    }
    /// An image memory barrier
    pub fn image_access(
        &mut self,
        image: (),
        stage: vk::PipelineStageFlags2, // The pipeline stage in which the image was accessed
        access: vk::AccessFlags2,       // The access type of the image
        src_layout: vk::ImageLayout,    // The image would be guaranteed to be in this layout before the node executes
        dst_layout: vk::ImageLayout,    // The image will be transitioned to this layout by this node
        subresource_range: vk::ImageSubresourceRange
    ) {

    }
    /// A global memory barrier
    pub fn access(
        &mut self,
        resource: (),
        stage: vk::PipelineStageFlags2, // The pipeline stage in which the image was accessed
        access: vk::AccessFlags2,       // The access type of the image
    ) {

    }

    /// Note that BufferMemoryBarriers don't actually do anything useful beyond global memory barrier.
    /// That's why we always emit global memory barrier for buffer resources, and don't ask for buffer offset and size.
    pub fn buffer_access(
        &mut self,
        resource: (),
        stage: vk::PipelineStageFlags2, // The pipeline stage in which the image was accessed
        access: vk::AccessFlags2,       // The access type of the image
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
        }
    }
}
impl RenderGraphContext {
    fn new() -> Self {
        Self {}
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
