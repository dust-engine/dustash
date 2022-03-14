use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use std::ops::Deref;
use std::sync::Arc;

use crate::Device;

pub struct SwapchainLoader {
    loader: khr::Swapchain,
    device: Arc<Device>,
}

impl SwapchainLoader {
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Deref for SwapchainLoader {
    type Target = khr::Swapchain;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}

impl SwapchainLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::Swapchain::new(device.instance(), &device);
        Self { loader, device }
    }
}

pub struct Swapchain {
    loader: Arc<SwapchainLoader>,
    pub(crate) swapchain: vk::SwapchainKHR,
}

impl Swapchain {
    pub unsafe fn create(
        loader: Arc<SwapchainLoader>,
        info: &vk::SwapchainCreateInfoKHR,
    ) -> VkResult<Self> {
        let swapchain = loader.create_swapchain(info, None)?;
        Ok(Self { loader, swapchain })
    }
    /// Returns (image_index, suboptimal)
    /// Semaphore must be binary semaphore
    pub unsafe fn acquire_next_image(
        &self,
        timeout_ns: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> VkResult<(u32, bool)> {
        // Requires exclusive access to swapchain
        self.loader
            .acquire_next_image(self.swapchain, timeout_ns, semaphore, fence)
    }

    pub unsafe fn get_swapchain_images(&self) -> VkResult<Vec<vk::Image>> {
        self.loader.get_swapchain_images(self.swapchain)
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
