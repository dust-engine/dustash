use crate::fence::Fence;
use crate::queue::semaphore::Semaphore;
use crate::queue::{QueueType, QueuesCreateInfo};
use crate::Device;
use crate::{resources::HasImage, swapchain::Swapchain};
use ash::{prelude::VkResult, vk};
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::{surface::Surface, swapchain::SwapchainLoader};

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
    extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,
    needs_rebuild: bool,

    generation: u64,
    old_swapchains: VecDeque<(vk::SwapchainKHR, u64)>,
}

impl FrameManager {
    pub fn device(&self) -> &Arc<Device> {
        self.swapchain_loader.device()
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
            .map(|_| unsafe {
                Ok(Frame {
                    complete_fence: Arc::new(Fence::new(swapchain_loader.device().clone(), true)?),
                    acquire_semaphore: Arc::new(
                        Semaphore::new(swapchain_loader.device().clone()).unwrap(),
                    ),
                    complete_semaphore: Arc::new(
                        Semaphore::new(swapchain_loader.device().clone()).unwrap(),
                    ),
                    generation: 0,
                })
            })
            .collect::<VkResult<Vec<Frame>>>()?;

        let queue_info = QueuesCreateInfo::find(swapchain_loader.device().physical_device());
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
                .expect("Can't find a queue family supporting presentation on this surface");
        let result = Self {
            swapchain_loader,
            swapchain: None,

            frames,
            frame_index: 0,
            surface,
            images: Vec::new(),
            extent,
            format: vk::SurfaceFormatKHR::default(),
            needs_rebuild: true,
            present_queue_family,

            options,
            generation: 0,
            old_swapchains: VecDeque::new(), // TODO: max capacity?
        };
        Ok(result)
    }

    pub fn update(&mut self, extent: vk::Extent2D) {
        self.extent = extent;
        self.needs_rebuild = true;
    }

    pub fn acquire(&mut self, timeout_ns: u64) -> VkResult<AcquiredFrame> {
        let next_frame_index = (self.frame_index + 1) % self.frames.len();
        unsafe {
            self.device().wait_for_fences(
                &[self.current_frame().complete_fence.fence],
                true,
                timeout_ns,
            )?;
            self.device()
                .reset_fences(&[self.current_frame().complete_fence.fence])?;
            // self.current_frame() has finished rendering
            while let Some(&(swapchain, generation)) = self.old_swapchains.front() {
                if self.frames[next_frame_index].generation == generation {
                    // next frame is still being rendered, and it's using the same swapchain as this one.
                    break;
                }
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
                            self.needs_rebuild = suboptimal;
                            let _invalidate_images =
                                self.current_frame().generation != self.generation; // TODO: ?
                            self.current_frame_mut().generation = self.generation;

                            let acquired_frame = AcquiredFrame {
                                image_index: index,
                                frame_index: self.frame_index,
                                ready_semaphore: self.current_frame().acquire_semaphore.clone(),
                                complete_semaphore: self.current_frame().complete_semaphore.clone(),
                                complete_fence: self.current_frame().complete_fence.clone(),
                                image: self.images[index as usize],
                                present_queue_family: self.present_queue_family,
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
        let surface_capabilities = self
            .surface
            .get_capabilities(self.device().physical_device())?;
        self.extent = match surface_capabilities.current_extent.width {
            // If Vulkan doesn't know, the windowing system probably does. Known to apply at
            // least to Wayland.
            std::u32::MAX => vk::Extent2D {
                width: self.extent.width,
                height: self.extent.height,
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_mode = self
            .surface
            .get_present_modes(self.device().physical_device())?
            .iter()
            .filter_map(|&mode| {
                Some((
                    mode,
                    self.options
                        .present_mode_preference
                        .iter()
                        .position(|&pref| pref == mode)?,
                ))
            })
            .min_by_key(|&(_, priority)| priority)
            .map(|(mode, _)| mode)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

        let image_count = (self.frames.len() as u32)
            .min(surface_capabilities.max_image_count)
            .max(surface_capabilities.min_image_count);
        self.format = self
            .surface
            .get_formats(self.device().physical_device())?
            .iter()
            .filter_map(|&format| {
                Some((
                    format,
                    self.options
                        .format_preference
                        .iter()
                        .position(|&pref| pref == format)?,
                ))
            })
            .min_by_key(|&(_, priority)| priority)
            .map(|(mode, _)| mode)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

        let old_swapchain = self.swapchain.take();
        // take apart old_swapchain by force
        let (swapchain_loader, old_swapchain) = if let Some(old_swapchain) = old_swapchain {
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
            (self.swapchain_loader.clone(), vk::SwapchainKHR::null())
        };

        let info = vk::SwapchainCreateInfoKHR {
            surface: self.surface.surface,
            min_image_count: image_count,
            image_color_space: self.format.color_space,
            image_format: self.format.format,
            image_extent: self.extent,
            image_usage: self.options.usage,
            image_sharing_mode: self.options.sharing_mode,
            pre_transform,
            composite_alpha: self.options.composite_alpha,
            present_mode,
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
        self.swapchain = Some(new_swapchain);
        self.needs_rebuild = false;

        Ok(())
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
    complete_fence: Arc<Fence>,
    acquire_semaphore: Arc<Semaphore>,
    complete_semaphore: Arc<Semaphore>,
    generation: u64,
}

#[derive(Clone, Debug)]
pub struct AcquiredFrame {
    /// Queue family to present on
    pub present_queue_family: u32,

    /// Index of the image to write to in [`Swapchain::images`]
    pub image_index: u32,
    /// Index of the frame in flight, for use tracking your own per-frame resources, which may be
    /// accessed immediately after [`Swapchain::acquire`] returns
    pub frame_index: usize,
    /// Must be waited on before accessing the image associated with `image_index`
    pub ready_semaphore: Arc<Semaphore>,

    /// Must be signaled when access is complete
    pub complete_semaphore: Arc<Semaphore>,

    /// Must be signaled when access to the image associated with `image_index` and any per-frame
    /// resources associated with `frame_index` is complete
    pub complete_fence: Arc<Fence>,

    pub image: vk::Image, // Always valid, since we retain a reference to the swapchain
}

impl HasImage for AcquiredFrame {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
}
