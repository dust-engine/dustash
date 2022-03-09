use std::{future::Future, sync::Arc};

use ash::{prelude::VkResult, vk};

use crate::Device;

pub struct Semaphore {
    pub(crate) device: Arc<Device>,
    pub(crate) semaphore: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: Arc<Device>) -> VkResult<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { device.create_semaphore(&create_info, None)? };
        Ok(Self { device, semaphore })
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        // Safety: Host access to semaphore must be externally synchronized
        // We have &mut self thus exclusive access to self.semaphore
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

#[repr(transparent)]
pub struct TimelineSemaphore(Semaphore);

impl TimelineSemaphore {
    pub fn new(device: Arc<Device>, initial_value: u64) -> VkResult<Self> {
        let type_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value)
            .build();
        let mut create_info = vk::SemaphoreCreateInfo::default();
        create_info.p_next = &type_info as *const _ as *const std::ffi::c_void;
        let semaphore = unsafe { device.create_semaphore(&create_info, None)? };
        Ok(TimelineSemaphore(Semaphore { device, semaphore }))
    }
    pub fn wait(self: Arc<TimelineSemaphore>, value: u64) -> impl Future<Output = VkResult<()>> {
        blocking::unblock(move || {
            let mut info = vk::SemaphoreWaitInfo::default();
            info.semaphore_count = 1;
            info.p_semaphores = &self.0.semaphore;
            info.p_values = &value;
            unsafe {
                self.0.device.wait_semaphores(&info, std::u64::MAX)?;
            }
            drop(self);
            Ok(())
        })
    }

    pub fn wait_n<const N: usize>(
        semaphores: [(Arc<TimelineSemaphore>, u64); N],
    ) -> impl Future<Output = VkResult<()>> {
        let device = semaphores[0].0 .0.device.clone();
        for (semaphore, _) in semaphores.iter() {
            assert_eq!(semaphore.0.device.handle(), device.handle());
        }

        blocking::unblock(move || {
            let raw_semaphores: [vk::Semaphore; N] =
                semaphores.each_ref().map(|s| s.0 .0.semaphore);
            let nums: [u64; N] = semaphores.each_ref().map(|s| s.1);

            let mut info = vk::SemaphoreWaitInfo::default();
            info.semaphore_count = N.try_into().unwrap();
            info.p_semaphores = raw_semaphores.as_ptr();
            info.p_values = nums.as_ptr();
            unsafe {
                device.wait_semaphores(&info, std::u64::MAX)?;
            }
            drop(semaphores);
            drop(device);
            Ok(())
        })
    }

    pub fn wait_many(
        semaphores: Vec<(Arc<TimelineSemaphore>, u64)>,
    ) -> impl Future<Output = VkResult<()>> {
        let device = semaphores[0].0 .0.device.clone();
        for (semaphore, _) in semaphores.iter() {
            assert_eq!(semaphore.0.device.handle(), device.handle());
        }

        blocking::unblock(move || {
            let raw_semaphores: Vec<vk::Semaphore> =
                semaphores.iter().map(|s| s.0 .0.semaphore).collect();
            let nums: Vec<u64> = semaphores.iter().map(|s| s.1).collect();

            let mut info = vk::SemaphoreWaitInfo::default();
            info.semaphore_count = semaphores.len() as u32;
            info.p_semaphores = raw_semaphores.as_ptr();
            info.p_values = nums.as_ptr();
            unsafe {
                device.wait_semaphores(&info, std::u64::MAX)?;
            }
            drop(semaphores);
            drop(device);
            Ok(())
        })
    }

    /// Downgrade an Arc<TimelineSemaphore> into an Arc<Semaphore>.
    pub fn as_arc_semaphore(self: Arc<TimelineSemaphore>) -> Arc<Semaphore> {
        unsafe {
            // Safety: This relies on TimelineSemaphore being transmutable to Semaphore.
            std::mem::transmute(self)
        }
    }
}
