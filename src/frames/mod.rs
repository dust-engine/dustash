use crate::queue::semaphore::{Semaphore, TimelineSemaphoreOp};
use crate::queue::{QueueType, QueuesCreateInfo};
use crate::Device;
use crate::{resources::HasImage, swapchain::Swapchain};
use ash::{prelude::VkResult, vk};
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::HasDevice;
use crate::{surface::Surface, swapchain::SwapchainLoader};

mod resource;
pub use resource::PerFrame;
mod uniform;
pub use uniform::PerFrameUniform;

/// Manages synchronizing and rebuilding a Vulkan swapchain
pub struct FrameManager {
    swapchain_loader: Arc<SwapchainLoader>,
    pub(crate) swapchain: Option<Swapchain>,
    surface: Arc<Surface>,
    present_queue_family: u32,

    options: Options,

    frames: Vec<Frame>,
    frame_index: usize,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    pre_transform: vk::SurfaceTransformFlagsKHR,
    image_count: u32,

    needs_rebuild: bool,

    generation: u64,
    old_swapchains: VecDeque<(vk::SwapchainKHR, u64)>,
}

impl crate::HasDevice for FrameManager {
    fn device(&self) -> &Arc<Device> {
        self.swapchain_loader.device()
    }
}

impl FrameManager {
    pub fn num_images(&self) -> usize {
        self.images.len()
    }
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    fn current_frame(&self) -> &Frame {
        &self.frames[self.frame_index]
    }
    fn current_frame_mut(&mut self) -> &mut Frame {
        &mut self.frames[self.frame_index]
    }
    /// Construct a new [`Swapchain`] for rendering at most `frames_in_flight` frames
    /// concurrently. `extent` should be the current dimensions of `surface`.
    pub fn new(
        swapchain_loader: Arc<SwapchainLoader>,
        surface: Arc<Surface>,
        options: Options,
        extent: vk::Extent2D,
    ) -> VkResult<Self> {
        assert_eq!(
            surface.loader().instance().handle(),
            swapchain_loader.device().instance().handle()
        );
        let frames = (0..options.frames_in_flight)
            .map(|_| {
                Ok(Frame {
                    acquire_semaphore: Arc::new(
                        Semaphore::new(swapchain_loader.device().clone()).unwrap(),
                    ),
                    complete_semaphore: Vec::new(),
                    generation: 0,
                    complete_timeline_semaphore: Vec::new(),
                })
            })
            .collect::<VkResult<Vec<Frame>>>()?;

        let queue_info = QueuesCreateInfo::find(swapchain_loader.device().physical_device());

        // Pick a present queue family.
        // This picks the first queue family supporting presentation on the surface in the given order:
        // - The Graphics family
        // - The Compute family
        // - All other families in the order that they were returned from the driver
        let present_queue_family =
            std::iter::once(queue_info.queue_family_index_for_type(QueueType::Graphics))
                .chain(std::iter::once(
                    queue_info.queue_family_index_for_type(QueueType::Compute),
                ))
                .chain(0_u32..queue_info.create_infos.len() as u32)
                .find(|&queue_family_index| {
                    surface
                        .supports_queue_family(
                            swapchain_loader.device().physical_device(),
                            queue_family_index,
                        )
                        .unwrap_or(false)
                })
                .expect("The physical device does not support presentation on this surface.");

        let surface_capabilities =
            surface.get_capabilities(swapchain_loader.device().physical_device())?;
        if !surface_capabilities
            .supported_usage_flags
            .contains(options.usage)
        {
            panic!()
        }
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_mode = surface
            .get_present_modes(swapchain_loader.device().physical_device())?
            .iter()
            .filter_map(|&mode| {
                Some((
                    mode,
                    options
                        .present_mode_preference
                        .iter()
                        .position(|&pref| pref == mode)?,
                ))
            })
            .min_by_key(|&(_, priority)| priority)
            .map(|(mode, _)| mode)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

        let image_count = (frames.len() as u32)
            // FIXME: max_image_count may be zero. Need to revisit the logic to select image count.
            .min(surface_capabilities.max_image_count)
            .max(surface_capabilities.min_image_count);
        let format = surface
            .pick_format(swapchain_loader.physical_device(), options.usage)
            .unwrap()
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

        let result = Self {
            swapchain_loader,
            swapchain: None,

            frames,
            frame_index: 0,
            surface,
            images: Vec::new(),
            image_views: Vec::new(),
            extent,
            format,
            needs_rebuild: true,
            present_queue_family,

            options,
            generation: 0,
            old_swapchains: VecDeque::new(), // TODO: max capacity?
            present_mode,
            pre_transform,
            image_count,
        };
        Ok(result)
    }

    pub fn update(&mut self, extent: vk::Extent2D) {
        self.extent = extent;
        self.needs_rebuild = true;
    }

    pub fn acquire(&mut self, timeout_ns: u64) -> VkResult<AcquiredFrame> {
        let span = tracing::info_span!("swapchain acquire");
        let _enter = span.enter();

        let next_frame_index = (self.frame_index + 1) % self.frames.len();
        unsafe {
            let semaphore_refs: Vec<_> = self
                .current_frame()
                .complete_timeline_semaphore
                .iter()
                .collect();
            TimelineSemaphoreOp::block_many(semaphore_refs.as_slice()).unwrap();
            // self.current_frame() has finished rendering
            while let Some(&(swapchain, generation)) = self.old_swapchains.front() {
                if self.frames[next_frame_index].generation == generation {
                    // next frame is still being rendered, and it's using the same swapchain as this one.
                    break;
                }
                tracing::event!(tracing::Level::DEBUG, ?swapchain, %generation, "destroy swapchain");
                self.swapchain_loader.destroy_swapchain(swapchain, None);
                self.old_swapchains.pop_front();
            }
            loop {
                if !self.needs_rebuild {
                    // self.swapchain is now guaranteed to be Some, since self.needs_rebuild was set to be true at initialization.
                    // Safety:
                    // - Host access to swapchain must be externally synchronized. We have &mut and thus ownership over self.swapchain.
                    // - Host access to semaphore must be externally synchronized. We have &mut and thus ownership over self.current_frame().acquire_semaphore.
                    // - Host access to fence must be externally syncronized. Fence is VK_NULL.
                    match self.swapchain.as_ref().unwrap().acquire_next_image(
                        timeout_ns,
                        self.current_frame().acquire_semaphore.semaphore,
                        vk::Fence::null(),
                    ) {
                        Ok((index, suboptimal)) => {
                            if suboptimal {
                                tracing::warn!("suboptimal image");
                                self.needs_rebuild = true;
                            }
                            self.needs_rebuild = suboptimal;
                            let invalidate_images =
                                self.current_frame().generation != self.generation;
                            self.current_frame_mut().generation = self.generation;

                            let acquired_frame = AcquiredFrame {
                                image_index: index,
                                frame_index: self.frame_index,
                                acquire_ready_semaphore: self
                                    .current_frame()
                                    .acquire_semaphore
                                    .clone(),
                                render_complete_semaphore_pool: std::mem::take(
                                    &mut self.current_frame_mut().complete_semaphore,
                                ),
                                render_complete_semaphores: Vec::new(),
                                image: self.images[index as usize],
                                image_view: self.image_views[index as usize],
                                present_queue_family: self.present_queue_family,
                                image_extent: self.extent,
                                invalidate_images,
                            };
                            self.frame_index = next_frame_index;
                            return Ok(acquired_frame);
                        }
                        // If outdated, acquire again
                        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {}
                        Err(e) => return Err(e),
                    }
                }
                self.needs_rebuild = true;
                self.rebuild_swapchain()?;
            }
        }
    }

    unsafe fn rebuild_swapchain(&mut self) -> VkResult<()> {
        let span = tracing::info_span!("rebuild swapchain");
        let _enter = span.enter();

        let old_swapchain = self.swapchain.take();
        // take apart old_swapchain by force
        let (swapchain_loader, old_swapchain) = if let Some(old_swapchain) = old_swapchain {
            tracing::debug!("recreate old swapchain");
            let swapchain = old_swapchain.swapchain;
            let mut old_swapchain = MaybeUninit::new(old_swapchain);
            let loader = {
                let mut loader: MaybeUninit<Arc<SwapchainLoader>> = MaybeUninit::uninit();
                std::mem::swap(
                    loader.assume_init_mut(),
                    &mut old_swapchain.assume_init_mut().loader,
                );
                // now old_swapchain is garbage
                drop(old_swapchain);
                loader.assume_init()
            };
            (loader, swapchain)
        } else {
            tracing::debug!("create new swapchain");
            (self.swapchain_loader.clone(), vk::SwapchainKHR::null())
        };

        let info = vk::SwapchainCreateInfoKHR {
            surface: self.surface.surface,
            min_image_count: self.image_count,
            image_color_space: self.format.color_space,
            image_format: self.format.format,
            image_extent: self.extent,
            image_usage: self.options.usage,
            image_sharing_mode: self.options.sharing_mode,
            pre_transform: self.pre_transform,
            composite_alpha: self.options.composite_alpha,
            present_mode: self.present_mode,
            clipped: vk::TRUE,
            image_array_layers: 1,
            old_swapchain,
            ..Default::default()
        };

        let new_swapchain = Swapchain::create(swapchain_loader, &info)?;

        // Can't destroy old swapchain yet.
        if old_swapchain != vk::SwapchainKHR::null() {
            self.old_swapchains
                .push_back((old_swapchain, self.generation))
        }
        self.generation = self.generation.wrapping_add(1);
        self.images = new_swapchain.get_swapchain_images()?;

        for view in self.image_views.drain(..) {
            self.swapchain_loader
                .device()
                .destroy_image_view(view, None);
        }
        self.image_views = self
            .images
            .iter()
            .map(|image| {
                self.device()
                    .create_image_view(
                        &vk::ImageViewCreateInfo {
                            image: *image,
                            view_type: vk::ImageViewType::TYPE_2D,
                            format: self.format.format,
                            components: vk::ComponentMapping {
                                r: vk::ComponentSwizzle::R,
                                g: vk::ComponentSwizzle::G,
                                b: vk::ComponentSwizzle::B,
                                a: vk::ComponentSwizzle::A,
                            },
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap()
            })
            .collect();
        self.swapchain = Some(new_swapchain);
        self.needs_rebuild = false;

        Ok(())
    }

    pub unsafe fn present(
        &mut self,
        present_queue: vk::Queue,
        mut frame: AcquiredFrame,
    ) -> VkResult<()> {
        let span =
            tracing::info_span!("present", ?present_queue, image_indice = ?frame.image_index);
        let _enter = span.enter();
        let semaphores: Vec<_> = frame
            .render_complete_semaphores
            .iter()
            .map(|s| s.0.semaphore)
            .collect();
        // frames.swapchain.is_some() is guaranteed to be true. frames.swapchain is only None on initialization, in which case we wouldn't have AcquiredFrame
        // Safety:
        // - Host access to queue must be externally synchronized. We have &mut self and thus ownership on present_queue.
        // - Host access to pPresentInfo->pWaitSemaphores[] must be externally synchronized. We have &mut frames, and frame.complete_semaphore
        // was borrowed from &mut frames. Therefore, we do have exclusive ownership on frame.complete_semaphore.
        // - Host access to pPresentInfo->pSwapchains[] must be externally synchronized. We have &mut frames, and thus ownership on frames.swapchain.
        let suboptimal = match self.swapchain.as_mut().unwrap().queue_present(
            present_queue,
            &semaphores,
            frame.image_index,
        ) {
            Ok(suboptimal) => suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(err) => return Err(err),
        };
        if suboptimal {
            tracing::warn!("suboptimal");
            self.needs_rebuild = true;
        }

        let (mut render_complete_semaphores, mut render_complete_timeline_semaphores) =
            frame.render_complete_semaphores.into_iter().unzip();
        // Recycle unused binary semaphores.
        self.frames[frame.frame_index]
            .complete_semaphore
            .append(&mut frame.render_complete_semaphore_pool);
        // Recycle used binary semaphores.
        self.frames[frame.frame_index]
            .complete_semaphore
            .append(&mut render_complete_semaphores);
        // Record timeline semaphores to wait
        self.frames[frame.frame_index]
            .complete_timeline_semaphore
            .append(&mut render_complete_timeline_semaphores);
        Ok(())
    }
}

impl Drop for FrameManager {
    fn drop(&mut self) {
        for view in self.image_views.drain(..) {
            unsafe {
                self.swapchain_loader
                    .device()
                    .destroy_image_view(view, None);
            }
        }
    }
}

/// [`Swapchain`] configuration
#[derive(Debug, Clone)]
pub struct Options {
    pub frames_in_flight: usize,
    pub format_preference: Vec<vk::SurfaceFormatKHR>,
    pub present_mode_preference: Vec<vk::PresentModeKHR>,
    pub usage: vk::ImageUsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            frames_in_flight: 3,
            format_preference: vec![
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_SRGB,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                },
                vk::SurfaceFormatKHR {
                    format: vk::Format::R8G8B8A8_SRGB,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                },
            ],
            present_mode_preference: vec![vk::PresentModeKHR::FIFO],
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        }
    }
}

struct Frame {
    acquire_semaphore: Arc<Semaphore>,
    complete_semaphore: Vec<Arc<Semaphore>>,
    complete_timeline_semaphore: Vec<TimelineSemaphoreOp>,
    generation: u64,
}

#[derive(Debug)]
pub struct AcquiredFrame {
    /// Queue family to present on
    pub present_queue_family: u32,

    /// Index of the image to write to.
    pub image_index: u32,
    /// Index of the frame in flight, for use tracking your own per-frame resources, which may be
    /// accessed immediately after [`FrameManager::acquire`] returns
    pub frame_index: usize,
    /// Must be waited on before accessing the image associated with `image_index`
    pub acquire_ready_semaphore: Arc<Semaphore>,

    /// List of available binary semaphores reused from previous frames.
    render_complete_semaphore_pool: Vec<Arc<Semaphore>>,

    /// List of binary semaphores to be awaited by vkQueuePresent.
    /// List of timeline semaphores signaled when rendering to swapchain is completed.
    pub(crate) render_complete_semaphores: Vec<(Arc<Semaphore>, TimelineSemaphoreOp)>,

    pub image: vk::Image, // Always valid, since we retain a reference to the swapchain
    pub image_view: vk::ImageView,
    pub image_extent: vk::Extent2D,

    /// If true, the image contained in this frame is different from previous frames.
    /// The application must re-record any command buffers
    pub invalidate_images: bool,
}

impl AcquiredFrame {
    pub fn get_render_complete_semaphore(&mut self) -> Arc<Semaphore> {
        let semaphore = self
            .render_complete_semaphore_pool
            .pop()
            .unwrap_or_else(|| {
                let semaphore = Semaphore::new(self.device().clone()).unwrap();
                Arc::new(semaphore)
            });
        semaphore
    }
}

impl HasImage for AcquiredFrame {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
}
impl crate::HasDevice for AcquiredFrame {
    fn device(&self) -> &Arc<Device> {
        self.acquire_ready_semaphore.device()
    }
}
