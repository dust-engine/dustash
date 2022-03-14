use crate::queue::{QueueType, QueuesCreateInfo};
use crate::Device;
use crate::{resources::HasImage, swapchain::Swapchain};
use ash::{prelude::VkResult, vk};
use std::sync::Arc;

use crate::{surface::Surface, swapchain::SwapchainLoader};

/// Manages synchronizing and rebuilding a Vulkan swapchain
pub struct FrameManager {
    swapchain_loader: Arc<SwapchainLoader>,
    swapchain: Option<Arc<Swapchain>>,
    surface: Arc<Surface>,
    present_queue_family: u32,

    options: Options,

    frames: Vec<Frame>,
    frame_index: usize,

    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,
    needs_rebuild: bool,
}

impl FrameManager {
    pub fn device(&self) -> &Arc<Device> {
        self.swapchain_loader.device()
    }

    fn current_frame(&self) -> &Frame {
        &self.frames[self.frame_index]
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
                    complete: swapchain_loader.device().create_fence(
                        &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )?,
                    acquire: swapchain_loader
                        .device()
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
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
        };
        Ok(result)
    }

    pub fn acquire(&mut self, timeout_ns: u64) -> VkResult<AcquiredFrame> {
        unsafe {
            self.device()
                .wait_for_fences(&[self.current_frame().complete], true, timeout_ns)?;
            self.device()
                .reset_fences(&[self.current_frame().complete])?;
            // TODO: drop reference to old swapchain
            loop {
                if !self.needs_rebuild {
                    // self.swapchain is now guaranteed to be Some, since self.needs_rebuild was set to be true at initialization.
                    match self.swapchain.as_ref().unwrap().acquire_next_image(
                        timeout_ns,
                        self.current_frame().acquire,
                        vk::Fence::null(),
                    ) {
                        Ok((index, suboptimal)) => {
                            self.needs_rebuild = suboptimal;
                            let acquired_frame = AcquiredFrame {
                                image_index: index as usize,
                                frame_index: self.frame_index,
                                ready: self.current_frame().acquire,
                                complete: self.current_frame().complete,
                                swapchain: self.swapchain.as_ref().unwrap().clone(),
                                image: self.images[index as usize],
                            };
                            self.frame_index = (self.frame_index + 1) % self.frames.len();
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
            old_swapchain: old_swapchain.map_or(vk::SwapchainKHR::null(), |s| s.swapchain),
            ..Default::default()
        };
        let new_swapchain = Swapchain::create(self.swapchain_loader.clone(), &info)?;
        self.images = new_swapchain.get_swapchain_images()?;
        self.swapchain = Some(Arc::new(new_swapchain));
        self.needs_rebuild = false;
        Ok(())
    }
}

/// [`Swapchain`] configuration
#[derive(Debug, Clone)]
pub struct Options {
    frames_in_flight: usize,
    format_preference: Vec<vk::SurfaceFormatKHR>,
    present_mode_preference: Vec<vk::PresentModeKHR>,
    usage: vk::ImageUsageFlags,
    sharing_mode: vk::SharingMode,
    composite_alpha: vk::CompositeAlphaFlagsKHR,
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
    complete: vk::Fence,
    acquire: vk::Semaphore,
}

#[derive(Clone)]
pub struct AcquiredFrame {
    /// A reference to the swapchain from which the frame was acquired.
    swapchain: Arc<Swapchain>,

    /// Index of the image to write to in [`Swapchain::images`]
    pub image_index: usize,
    /// Index of the frame in flight, for use tracking your own per-frame resources, which may be
    /// accessed immediately after [`Swapchain::acquire`] returns
    pub frame_index: usize,
    /// Must be waited on before accessing the image associated with `image_index`
    pub ready: vk::Semaphore,
    /// Must be signaled when access to the image associated with `image_index` and any per-frame
    /// resources associated with `frame_index` is complete
    pub complete: vk::Fence,

    pub image: vk::Image, // Always valid, since we retain a reference to the swapchain
}

impl AcquiredFrame {
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }
}

impl HasImage for AcquiredFrame {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
}
