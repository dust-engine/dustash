use crate::{Instance, PhysicalDevice};
use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use std::ops::Deref;
use std::sync::Arc;

pub struct SurfaceLoader {
    instance: Arc<Instance>,
    loader: khr::Surface,
}

impl SurfaceLoader {
    pub fn new(instance: Arc<Instance>) -> Self {
        let loader = khr::Surface::new(instance.entry(), &instance);
        SurfaceLoader { instance, loader }
    }
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

impl Deref for SurfaceLoader {
    type Target = khr::Surface;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}

pub struct Surface {
    loader: Arc<SurfaceLoader>,
    pub(crate) surface: vk::SurfaceKHR,
}

impl Surface {
    pub fn loader(&self) -> &Arc<SurfaceLoader> {
        &self.loader
    }
}

impl Surface {
    pub fn create(
        loader: Arc<SurfaceLoader>,
        window_handle: &dyn raw_window_handle::HasRawWindowHandle,
    ) -> VkResult<Surface> {
        let surface = unsafe {
            ash_window::create_surface(
                loader.instance.entry(),
                &loader.instance,
                window_handle,
                None,
            )?
        };
        Ok(Surface { loader, surface })
    }

    /// Query the basic capabilities of a surface, needed in order to create a swapchain
    pub fn get_capabilities(
        &self,
        pdevice: &PhysicalDevice,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        assert_eq!(pdevice.instance().handle(), self.loader.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(pdevice.raw(), self.surface)
        }
    }

    /// Determine whether a queue family of a physical device supports presentation to a given surface
    pub fn supports_queue_family(
        &self,
        pdevice: &PhysicalDevice,
        queue_family_index: u32,
    ) -> VkResult<bool> {
        assert_eq!(pdevice.instance().handle(), self.loader.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.loader.get_physical_device_surface_support(
                pdevice.raw(),
                queue_family_index,
                self.surface,
            )
        }
    }

    /// Query color formats supported by surface
    pub fn get_formats(&self, pdevice: &PhysicalDevice) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        assert_eq!(pdevice.instance().handle(), self.loader.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.loader
                .get_physical_device_surface_formats(pdevice.raw(), self.surface)
        }
    }

    /// Query color formats supported by surface
    pub fn get_present_modes(&self, pdevice: &PhysicalDevice) -> VkResult<Vec<vk::PresentModeKHR>> {
        assert_eq!(pdevice.instance().handle(), self.loader.instance.handle(), "Both of physicalDevice, and surface must have been created, allocated, or retrieved from the same VkInstance");
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes(pdevice.raw(), self.surface)
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        tracing::info!(surface = ?self.surface, "drop surface");
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}
