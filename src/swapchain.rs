use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use std::mem::MaybeUninit;
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
    pub(crate) loader: Arc<SwapchainLoader>,
    pub(crate) swapchain: vk::SwapchainKHR,
}

impl Swapchain {
    /// # Safety
    /// https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html
    pub unsafe fn create(
        loader: Arc<SwapchainLoader>,
        info: &vk::SwapchainCreateInfoKHR,
    ) -> VkResult<Self> {
        let swapchain = loader.create_swapchain(info, None)?;
        Ok(Self { loader, swapchain })
    }
    /// Returns (image_index, suboptimal)
    /// Semaphore must be binary semaphore
    /// # Safety
    /// https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkAcquireNextImageKHR.html
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

    // Returns: Suboptimal
    pub unsafe fn queue_present(
        &mut self,
        queue: vk::Queue,
        wait_semaphores: &[vk::Semaphore],
        image_indice: u32,
    ) -> VkResult<bool> {
        self.loader.queue_present(
            queue,
            &vk::PresentInfoKHR {
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: &self.swapchain,
                p_image_indices: &image_indice,
                p_results: std::ptr::null_mut(), // Applications that do not need per-swapchain results can use NULL for pResults.
                ..Default::default()
            },
        )
    }

    pub fn get_swapchain_images(&self) -> VkResult<Vec<vk::Image>> {
        unsafe { self.loader.get_swapchain_images(self.swapchain) }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        tracing::info!(swapchain = ?self.swapchain, "drop swapchain");
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
