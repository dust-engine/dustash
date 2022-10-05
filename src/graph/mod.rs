use ash::vk;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::rc::Weak;
use std::sync::Arc;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::accel_struct::AccelerationStructure;
use crate::command::recorder::{CommandBufferResource, CommandRecorder, ReferencedResource};
use crate::descriptor::{DescriptorPool, DescriptorSet};
use crate::pipeline::{Binding, Pipeline, PipelineCache};
use crate::resources::buffer::HasBufferView;
use crate::resources::image::HasImageView;
use crate::resources::{HasBuffer, HasImage};

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
impl<T> Eq for BinaryHeapKeyedEntry<T> {}

struct RenderGraphNode {
    nexts: Vec<BinaryHeapKeyedEntry<Rc<RefCell<RenderGraphNode>>>>,
    config: RenderGraphContext,
}

pub struct Then {
    heads: Vec<Weak<RefCell<RenderGraphNode>>>,
}

pub struct ResourceHandle<T> {
    idx: usize,
    _marker: PhantomData<T>,
}
impl<T> Clone for ResourceHandle<T> {
    fn clone(&self) -> Self {
        Self {
            idx: self.idx,
            _marker: PhantomData,
        }
    }
}
impl<T> Copy for ResourceHandle<T> {}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            heads: BinaryHeap::new(),
            resources: Vec::new(),
        }
    }
    pub fn start<F: FnOnce(&mut RenderGraphContext) + 'static>(&mut self, run: F) -> Then {
        let mut config = RenderGraphContext::new();
        run(&mut config);
        let node = BinaryHeapKeyedEntry(
            config.priority,
            Rc::new(RefCell::new(RenderGraphNode {
                nexts: Vec::new(),
                config,
            })),
        );
        let head = Rc::downgrade(&node.1);
        self.heads.push(node);
        Then { heads: vec![head] }
    }

    pub fn import<T: Send + Sync + 'static>(&mut self, resource: T) -> ResourceHandle<T> {
        let idx = self.resources.len();
        self.resources.push(Resource::Other(Box::new(resource)));
        ResourceHandle {
            idx,
            _marker: PhantomData,
        }
    }

    pub fn import_image<T: HasImage + Send + Sync + 'static>(
        &mut self,
        resource: T,
    ) -> ResourceHandle<T> {
        let idx = self.resources.len();
        self.resources.push(Resource::Image(Box::new(resource)));
        ResourceHandle {
            idx,
            _marker: PhantomData,
        }
    }

    pub fn import_buffer<T: HasBuffer + Send + Sync + 'static>(
        &mut self,
        resource: T,
    ) -> ResourceHandle<T> {
        let idx = self.resources.len();
        self.resources.push(Resource::Buffer(Box::new(resource)));
        ResourceHandle {
            idx,
            _marker: PhantomData,
        }
    }
    pub fn run(
        mut self,
        mut command_recorder: CommandRecorder,
        pipeline_cache: &mut PipelineCache,
    ) {
        // Pipeline resource state
        let mut resources = self
            .resources
            .into_iter()
            .map(|res| ResourceState {
                resource: res,
                dirty_stages: vk::PipelineStageFlags2::empty(),
                accesses: vk::AccessFlags2::empty(),
                prev_write: false,
                available_stages: vk::PipelineStageFlags2::empty(),
                available_accesses: vk::AccessFlags2::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            })
            .collect::<Vec<_>>();

        // Pipeline binding state
        let mut descriptor_cache: HashMap<
            BTreeMap<u32, (usize, RenderGraphContextBinding)>,
            Arc<DescriptorSet>,
        > = HashMap::new();
        let mut bindings: BTreeMap<u32, Arc<DescriptorSet>> = BTreeMap::new();
        let descriptor_pool: Arc<DescriptorPool> = (|| todo!())();

        while let Some(head) = self.heads.pop() {
            let mut head = match Rc::try_unwrap(head.1) {
                Ok(head) => head.into_inner(),
                Err(_) => panic!("Requries exclusive ownership"),
            };
            if let Some(record) = head.config.record {
                {
                    // Insert barriers
                    let mut global_barries: Vec<vk::MemoryBarrier2> = Vec::new();
                    let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
                    let mut buffer_barries: Vec<vk::BufferMemoryBarrier2> = Vec::new();

                    for access in head.config.accesses.iter() {
                        let res = &mut resources[access.idx];

                        let is_write =
                            crate::util::pipline_stage_order::access_is_write(access.access);

                        let mut add_barrier =
                            |src_access_mask: vk::AccessFlags2,
                             dst_access_mask: vk::AccessFlags2| {
                                match access.barrier {
                                    Barrier::Image {
                                        src_layout,
                                        dst_layout,
                                        subresource_range,
                                    } => {
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
                                                _ => panic!(),
                                            },
                                            ..Default::default()
                                        });
                                        res.layout = dst_layout;
                                    }
                                    Barrier::ImageView {
                                        src_layout,
                                        dst_layout,
                                    } => {
                                        let (image, subresource_range) = match &res.resource {
                                            Resource::ImageView(image_view) => (
                                                image_view.raw_image(),
                                                image_view.subresource_range(),
                                            ),
                                            _ => panic!(),
                                        };
                                        image_barriers.push(vk::ImageMemoryBarrier2 {
                                            src_stage_mask: res.dirty_stages,
                                            src_access_mask,
                                            dst_stage_mask: access.stage,
                                            dst_access_mask,
                                            old_layout: res.layout,
                                            new_layout: src_layout,
                                            subresource_range,
                                            image,
                                            ..Default::default()
                                        });
                                        res.layout = dst_layout;
                                    }
                                    Barrier::Global => global_barries.push(vk::MemoryBarrier2 {
                                        src_stage_mask: res.dirty_stages,
                                        src_access_mask,
                                        dst_stage_mask: access.stage,
                                        dst_access_mask,
                                        ..Default::default()
                                    }),
                                    Barrier::Buffer { offset, size } => {
                                        buffer_barries.push(vk::BufferMemoryBarrier2 {
                                            src_stage_mask: res.dirty_stages,
                                            src_access_mask,
                                            dst_stage_mask: access.stage,
                                            dst_access_mask,
                                            offset,
                                            size,
                                            buffer: match &res.resource {
                                                Resource::Buffer(image) => image.raw_buffer(),
                                                _ => panic!(),
                                            },
                                            ..Default::default()
                                        })
                                    }
                                    Barrier::BufferView => {
                                        let (buffer, offset, size) = match &res.resource {
                                            Resource::BufferView(buffer_view) => {
                                                let (offset, size) =
                                                    buffer_view.subresource_range();
                                                (buffer_view.raw_buffer(), offset, size)
                                            }
                                            _ => panic!(),
                                        };
                                        buffer_barries.push(vk::BufferMemoryBarrier2 {
                                            src_stage_mask: res.dirty_stages,
                                            src_access_mask,
                                            dst_stage_mask: access.stage,
                                            dst_access_mask,
                                            offset,
                                            size,
                                            buffer,
                                            ..Default::default()
                                        })
                                    }
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
                            }
                            (true, false) => {
                                // Read after write.
                                add_barrier(res.accesses, access.access);
                                res.available_stages =
                                    crate::util::pipline_stage_order::logically_later_stages(
                                        access.stage,
                                    );
                                res.available_accesses = access.access;
                            }
                            (false, true) => {
                                // Write after read
                                // Execution barrier only.
                                add_barrier(vk::AccessFlags2::empty(), vk::AccessFlags2::empty());
                                res.dirty_stages = access.stage;
                                res.accesses = access.access;
                                res.available_stages = vk::PipelineStageFlags2::empty();
                                res.available_accesses = vk::AccessFlags2::empty();
                            }
                            (false, false) => {
                                // Read after read. Only emit barrier when it's not already covered.
                                if !res.available_stages.contains(access.stage)
                                    || !res.available_accesses.contains(access.access)
                                {
                                    // Re-emit barrier.
                                    add_barrier(res.accesses, access.access);
                                    res.available_stages |=
                                        crate::util::pipline_stage_order::logically_later_stages(
                                            access.stage,
                                        );
                                    res.available_accesses |= access.access;
                                }
                            }
                        }
                        res.prev_write = is_write;
                    }
                    unsafe {
                        command_recorder.pipeline_barrier2(&vk::DependencyInfo {
                            dependency_flags: vk::DependencyFlags::BY_REGION,
                            memory_barrier_count: global_barries.len() as u32,
                            p_memory_barriers: global_barries.as_ptr(),
                            buffer_memory_barrier_count: buffer_barries.len() as u32,
                            p_buffer_memory_barriers: buffer_barries.as_ptr(),
                            image_memory_barrier_count: image_barriers.len() as u32,
                            p_image_memory_barriers: image_barriers.as_ptr(),
                            ..Default::default()
                        });
                    }
                }
                if let Some(pipeline) = &head.config.pipeline {
                    // Pipeline & Descriptor set
                    // Bind descriptor sets
                    let pipeline_layout = pipeline.layout();
                    for (set_id, set) in head.config.bindings.into_iter() {
                        let descriptor_set = descriptor_cache.entry(set).or_insert_with_key(|set| {
                            // Allocate new descriptor set
                            let (_, descriptor_set_layout) = &pipeline_layout.descriptor_sets[set_id as usize];
                            let desc_set = descriptor_pool.allocate_one(descriptor_set_layout).unwrap();


                            let mut p_image_infos: Vec<vk::DescriptorImageInfo> = Vec::with_capacity(set.len());
                            let mut p_buffer_info: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(set.len());
                            let mut p_texel_buffer_view: Vec<vk::BufferView> = Vec::with_capacity(set.len());
                            let mut p_accel_struct: Vec<vk::AccelerationStructureKHR> = Vec::with_capacity(set.len());
                            let mut p_accel_struct_info: Vec<vk::WriteDescriptorSetAccelerationStructureKHR> = Vec::with_capacity(set.len());


                            let writes: Vec<_> = set.iter().map(|(binding_id, (resource_id, binding))| {
                                let resource = &resources[*resource_id];
                                let mut base = vk::WriteDescriptorSet {
                                    dst_set: desc_set.raw(),
                                    dst_binding: *binding_id,
                                    dst_array_element: 0,
                                    descriptor_count: 1,
                                    ..Default::default()
                                };
                                match binding {
                                    RenderGraphContextBinding::Image { layout } => unsafe {
                                        let index = p_image_infos.len();
                                        assert_eq!(resource.layout, *layout);
                                        p_image_infos.push(vk::DescriptorImageInfo {
                                            sampler: vk::Sampler::null(),
                                            image_layout: *layout,
                                            image_view: match &resource.resource {
                                                Resource::ImageView(image_view) => image_view.raw_image_view(),
                                                _ => panic!()
                                            }
                                        });
                                        base.p_image_info = p_image_infos.as_ptr().add(index);
                                    },
                                    RenderGraphContextBinding::Buffer { offset, size } => unsafe {
                                        let index = p_buffer_info.len();
                                        p_buffer_info.push(vk::DescriptorBufferInfo {
                                            offset: *offset,
                                            range: *size,
                                            buffer: match &resource.resource {
                                                Resource::Buffer(buffer) => buffer.raw_buffer(),
                                                Resource::BufferView(buffer_view) => buffer_view.raw_buffer(),
                                                _ => panic!()
                                            }
                                        });
                                        base.p_buffer_info = p_buffer_info.as_ptr().add(index);
                                    },
                                    RenderGraphContextBinding::TexelBuffer => unsafe {
                                        let index = p_texel_buffer_view.len();
                                        p_texel_buffer_view.push(match &resource.resource {
                                            Resource::BufferView(buffer_view) => buffer_view.raw_buffer_view(),
                                            _ => panic!()
                                        });
                                        base.p_texel_buffer_view = p_texel_buffer_view.as_ptr().add(index);
                                    },
                                    RenderGraphContextBinding::AccelerationStructure => unsafe {
                                        let index = p_accel_struct.len();
                                        p_accel_struct.push(match &resource.resource {
                                            Resource::AccelerationStructure(accel_struct) => accel_struct.raw(),
                                            _ => panic!()
                                        });
                                        p_accel_struct_info.push(vk::WriteDescriptorSetAccelerationStructureKHR {
                                            acceleration_structure_count: 1,
                                            p_acceleration_structures: p_accel_struct.as_ptr().add(index),
                                            ..Default::default()
                                        });
                                        base.p_next = p_accel_struct_info.as_ptr().add(index) as *const _;
                                    },
                                }
                                base
                            }).collect();
                            unsafe {
                                command_recorder.device.update_descriptor_sets(writes.as_slice(), &[]);
                            }
                            Arc::new(desc_set)
                        });
                        if let Some(set) = bindings.get(&set_id) && Arc::ptr_eq(set, &descriptor_set) {
                            // The descriptor set was already binded to the correct slot. Nothing we need to do here.
                        } else {
                            // Bind descrptor set
                            unsafe {
                                command_recorder.bind_descriptor_set(
                                     pipeline.bind_point(),
                                    pipeline.layout(),
                                    set_id,
                                    &[descriptor_set.raw()],
                                    &[]
                                );
                            }
                            bindings.insert(set_id, descriptor_set.clone());
                        }
                    }
                }
                // Execute the command.
                let mut ctx = RenderGraphRecordingContext {
                    command_recorder: &mut command_recorder,
                    resources: resources.as_slice(),
                    pipeline: head.config.pipeline,
                };
                (record)(&mut ctx);
            }

            let new_heads = head
                .nexts
                .drain_filter(|next| Rc::strong_count(&mut next.1) == 1);
            self.heads.extend(new_heads);
        }
        command_recorder
            .referenced_resources
            .extend(resources.into_iter().map(|a| match a.resource {
                Resource::Other(res) => res.command_buffer_resource(),
                Resource::AccelerationStructure(res) => res.command_buffer_resource(),
                Resource::Buffer(buffer) => buffer.boxed_type_erased().command_buffer_resource(),
                Resource::Image(image) => image.boxed_type_erased().command_buffer_resource(),
                Resource::BufferView(buffer) => {
                    buffer.boxed_type_erased().command_buffer_resource()
                }
                Resource::ImageView(image) => image.boxed_type_erased().command_buffer_resource(),
            }));
    }
}

pub enum Resource {
    Buffer(Box<dyn HasBuffer + Send + Sync + 'static>),
    BufferView(Box<dyn HasBufferView + Send + Sync + 'static>),
    Image(Box<dyn HasImage + Send + Sync + 'static>),
    ImageView(Box<dyn HasImageView + Send + Sync + 'static>),
    AccelerationStructure(Arc<AccelerationStructure>),
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
    pub fn then<F: FnOnce(&mut RenderGraphContext) + 'static>(&self, run: F) -> Then {
        let mut config = RenderGraphContext::new();
        run(&mut config);
        let node = BinaryHeapKeyedEntry(
            config.priority,
            Rc::new(RefCell::new(RenderGraphNode {
                nexts: Vec::new(),
                config,
            })),
        );
        let head = Rc::downgrade(&node.1);
        for head in self.heads.iter() {
            head.upgrade()
                .unwrap() // This should still be referenced by the graph itself.
                .borrow_mut()
                .nexts
                .push(node.clone());
        }
        Then { heads: vec![head] }
    }
    pub fn join(&self, other: &Then) -> Then {
        let mut heads = Vec::with_capacity(self.heads.len() + other.heads.len());
        heads.extend_from_slice(&self.heads);
        heads.extend_from_slice(&other.heads);
        Then { heads }
    }
}

pub struct RenderGraphRecordingContext<'a, 'b> {
    command_recorder: &'a mut CommandRecorder<'b>,
    resources: &'a [ResourceState],
    pipeline: Option<Arc<dyn Pipeline>>,
}

impl<'a, 'b> RenderGraphRecordingContext<'a, 'b> {
    pub fn get_image<T: HasImage>(
        &self,
        handle: ResourceHandle<T>,
    ) -> &(dyn HasImage + Send + Sync) {
        match &self.resources[handle.idx].resource {
            Resource::Image(img) => img.as_ref(),
            _ => panic!("Error: Resource wasn't imported as an image"),
        }
    }
    pub fn get_buffer<T: HasBuffer>(
        &self,
        handle: ResourceHandle<T>,
    ) -> &(dyn HasBuffer + Send + Sync) {
        match &self.resources[handle.idx].resource {
            Resource::Buffer(buf) => buf.as_ref(),
            _ => panic!("Error: Resource wasn't imported as an image"),
        }
    }
    pub fn bind_pipeline(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            unsafe {
                self.command_recorder.device.cmd_bind_pipeline(
                    self.command_recorder.command_buffer,
                    pipeline.bind_point(),
                    pipeline.raw(),
                );
            }
            let pipeline: Arc<dyn Send + Sync> = pipeline.arc_type_erased();
            self.command_recorder
                .referenced_resources
                .push(pipeline.command_buffer_resource());
        }
    }
}

#[derive(PartialEq, Eq, Clone, Hash)]
enum RenderGraphContextBinding {
    Image {
        layout: vk::ImageLayout,
    },
    Buffer {
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    },
    TexelBuffer,
    AccelerationStructure,
}
pub struct RenderGraphContext {
    /// The priority of the task.
    ///
    /// The value of this priority could be fine tuned to ensure maximum overlapping.
    /// However, a simple heuristic applies  here:
    /// A node should have lower priority if it consumes many results that are expensive to produce.
    /// A node should have higher priority if it is expensive to execute and its products are consumed by many dependent nodes.
    priority: isize,
    record: Option<Box<dyn FnOnce(&mut RenderGraphRecordingContext)>>,
    accesses: Vec<Access>,

    // Mapping: set id -> binding id -> resource id
    bindings: BTreeMap<u32, BTreeMap<u32, (usize, RenderGraphContextBinding)>>,
    pipeline: Option<Arc<dyn Pipeline>>,
}
pub struct RenderGraphPipelineContext<'a, P: Pipeline> {
    inner: &'a mut RenderGraphContext,
    pipeline: Arc<P>,
}

impl RenderGraphContext {
    pub fn record(
        &mut self,
        record: impl FnOnce(&mut RenderGraphRecordingContext) + 'static,
    ) -> &mut Self {
        self.record = Some(Box::new(record));
        self
    }
    /// An image memory barrier
    pub fn image_access<T: HasImage>(
        &mut self,
        image: ResourceHandle<T>,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        // The image would be guaranteed to be in this layout before the node executes.
        // If the image was produced in a different layout by the previous node, the render graph will
        // insert pipeline barrier to transition the node to the src_layout before the node executes.
        src_layout: vk::ImageLayout,
        // The image will be transitioned to this layout by this node
        dst_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) -> &mut Self {
        if stage == vk::PipelineStageFlags2::empty() {
            return self;
        }
        self.accesses.push(Access {
            idx: image.idx,
            stage,
            access,
            barrier: Barrier::Image {
                src_layout,
                dst_layout,
                subresource_range,
            },
        });
        self
    }
    pub fn image_view_access<T: HasImage>(
        &mut self,
        image: ResourceHandle<T>,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        // The image would be guaranteed to be in this layout before the node executes.
        // If the image was produced in a different layout by the previous node, the render graph will
        // insert pipeline barrier to transition the node to the src_layout before the node executes.
        src_layout: vk::ImageLayout,
        // The image will be transitioned to this layout by this node
        dst_layout: vk::ImageLayout,
    ) -> &mut Self {
        if stage == vk::PipelineStageFlags2::empty() {
            return self;
        }
        self.accesses.push(Access {
            idx: image.idx,
            stage,
            access,
            barrier: Barrier::ImageView {
                src_layout,
                dst_layout,
            },
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
        if stage == vk::PipelineStageFlags2::empty() {
            return self;
        }
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
        if stage == vk::PipelineStageFlags2::empty() {
            return self;
        }
        self.accesses.push(Access {
            idx: resource.idx,
            stage,
            access,
            barrier: Barrier::Buffer { offset, size },
        });
        self
    }

    pub fn buffer_view_access<T: HasBuffer>(
        &mut self,
        resource: ResourceHandle<T>,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> &mut Self {
        if stage == vk::PipelineStageFlags2::empty() {
            return self;
        }
        self.accesses.push(Access {
            idx: resource.idx,
            stage,
            access,
            barrier: Barrier::BufferView,
        });
        self
    }
    pub fn pipeline<P: Pipeline + 'static>(
        &mut self,
        pipeline: Arc<P>,
    ) -> RenderGraphPipelineContext<P> {
        self.pipeline = Some(pipeline.clone());
        RenderGraphPipelineContext {
            inner: self,
            pipeline,
        }
    }

    /// Convenient method to copy buffer.
    /// Calls `buffer_access` and record the command buffer.
    pub fn copy_buffer<S: HasBuffer, T: HasBuffer>(
        &mut self,
        src: ResourceHandle<S>,
        dst: ResourceHandle<T>,
        region: vk::BufferCopy,
    ) {
        self.buffer_access(
            src,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
            region.src_offset,
            region.size,
        );
        self.buffer_access(
            dst,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
            region.dst_offset,
            region.size,
        );

        self.record(move |cb| unsafe {
            cb.command_recorder.device.cmd_copy_buffer(
                cb.command_recorder.command_buffer,
                cb.get_buffer(src).raw_buffer(),
                cb.get_buffer(dst).raw_buffer(),
                &[region],
            )
        });
    }
    /// Convenient method to copy more than one region in the same buffer.
    /// Calls `buffer_access` and record the command buffer.
    pub fn copy_buffer_batched<S: HasBuffer, T: HasBuffer>(
        &mut self,
        src: ResourceHandle<S>,
        dst: ResourceHandle<T>,
        regions: impl IntoIterator<Item = vk::BufferCopy>,
    ) {
        let regions: Vec<vk::BufferCopy> = regions.into_iter().collect();
        if regions.len() > 1 {
            // If copying multiple regions, use global barrier instead.
            // Buffer barriers don't actually do anything useful. No current GPUs can performance such fine grained syncronization.
            self.access(
                src,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            );
            self.access(
                dst,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            );
        } else {
            let region = &regions[0];
            self.buffer_access(
                src,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
                region.src_offset,
                region.size,
            );
            self.buffer_access(
                dst,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
                region.dst_offset,
                region.size,
            );
        }
        self.record(move |cb| unsafe {
            cb.command_recorder.device.cmd_copy_buffer(
                cb.command_recorder.command_buffer,
                cb.get_buffer(src).raw_buffer(),
                cb.get_buffer(dst).raw_buffer(),
                &regions,
            )
        });
    }

    fn new() -> Self {
        Self {
            priority: 0,
            record: None,
            accesses: Vec::new(),
            bindings: BTreeMap::new(),
            pipeline: None,
        }
    }
}

impl<'a, P: Pipeline> RenderGraphPipelineContext<'a, P> {
    fn bind<T>(
        &mut self,
        set_id: u32,
        binding_id: u32,
        resource: ResourceHandle<T>,
        binding: RenderGraphContextBinding,
    ) -> &mut Self {
        // Insert into self.inner.bindings
        let old_id = self
            .inner
            .bindings
            .entry(set_id)
            .or_default()
            .insert(binding_id, (resource.idx, binding));
        assert!(old_id.is_none(), "Duplicated binding");

        self
    }
    pub fn bind_buffer<T: HasBuffer>(
        &mut self,
        set_id: u32,
        binding_id: u32,
        resource: ResourceHandle<T>,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> &mut Self {
        self.bind(
            set_id,
            binding_id,
            resource,
            RenderGraphContextBinding::Buffer { offset, size },
        );
        let binding: &Binding = self
            .pipeline
            .binding(set_id, binding_id)
            .expect("Unknown binding");
        self.inner.buffer_access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_read_stage_flags),
            crate::util::descriptor_type_to_access_flags_read(binding.ty),
            offset,
            size,
        );
        self.inner.buffer_access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_write_stage_flags),
            crate::util::descriptor_type_to_access_flags_write(binding.ty),
            offset,
            size,
        );
        self
    }
    pub fn bind_image<T: HasImageView>(
        &mut self,
        set_id: u32,
        binding_id: u32,
        resource: ResourceHandle<T>,
        layout: vk::ImageLayout,
    ) -> &mut Self {
        self.bind(
            set_id,
            binding_id,
            resource,
            RenderGraphContextBinding::Image { layout },
        );
        let binding: &Binding = self
            .pipeline
            .binding(set_id, binding_id)
            .expect("Unknown binding");
        self.inner.image_view_access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_read_stage_flags),
            crate::util::descriptor_type_to_access_flags_read(binding.ty),
            layout,
            layout,
        );
        self.inner.image_view_access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_write_stage_flags),
            crate::util::descriptor_type_to_access_flags_write(binding.ty),
            layout,
            layout,
        );
        self
    }
    pub fn bind_texel_buffer<T: HasBufferView>(
        &mut self,
        set_id: u32,
        binding_id: u32,
        resource: ResourceHandle<T>,
    ) -> &mut Self {
        self.bind(
            set_id,
            binding_id,
            resource,
            RenderGraphContextBinding::TexelBuffer,
        );
        let binding: &Binding = self
            .pipeline
            .binding(set_id, binding_id)
            .expect("Unknown binding");
        self.inner.buffer_view_access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_read_stage_flags),
            crate::util::descriptor_type_to_access_flags_read(binding.ty),
        );
        self.inner.buffer_view_access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_write_stage_flags),
            crate::util::descriptor_type_to_access_flags_write(binding.ty),
        );
        self
    }
    pub fn bind_acceleration_structure(
        &mut self,
        set_id: u32,
        binding_id: u32,
        resource: ResourceHandle<Arc<AccelerationStructure>>,
    ) -> &mut Self {
        self.bind(
            set_id,
            binding_id,
            resource,
            RenderGraphContextBinding::AccelerationStructure,
        );
        let binding: &Binding = self
            .pipeline
            .binding(set_id, binding_id)
            .expect("Unknown binding");
        self.inner.access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_read_stage_flags),
            crate::util::descriptor_type_to_access_flags_read(binding.ty),
        );
        self.inner.access(
            resource,
            crate::util::shader_stage_to_pipeline_stage(binding.shader_write_stage_flags),
            crate::util::descriptor_type_to_access_flags_write(binding.ty),
        );
        self
    }
}

struct Access {
    idx: usize,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    barrier: Barrier,
}
enum Barrier {
    Image {
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    },
    ImageView {
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    },
    Global,
    Buffer {
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    },
    BufferView,
}

#[cfg(test)]
mod tests {
    use crate::ray_tracing::sbt::Sbt;

    use super::*;
    #[test]
    fn test() {
        let mut graph = RenderGraph::new();
        let sbt = Sbt::new(todo!(), [0], [0], [0], [(0, 0)], todo!(), &mut graph);

        graph.start(sbt.transfer());
    }
}
