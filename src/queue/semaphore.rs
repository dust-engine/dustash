use std::{future::Future, sync::Arc};

use ash::{prelude::VkResult, vk};

use crate::Device;

pub struct Semaphore {
    pub(crate) device: Arc<Device>,
    pub(crate) semaphore: vk::Semaphore,
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Semaphore({:?})", self.semaphore))
    }
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
        tracing::debug!(semaphore = ?self.semaphore, "drop semaphore");
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
        let create_info = vk::SemaphoreCreateInfo {
            p_next: &type_info as *const _ as *const std::ffi::c_void,
            ..Default::default()
        };
        let semaphore = unsafe { device.create_semaphore(&create_info, None)? };
        Ok(TimelineSemaphore(Semaphore { device, semaphore }))
    }
    pub fn signal(&self, value: u64) -> VkResult<()> {
        unsafe {
            self.0.device.signal_semaphore(&vk::SemaphoreSignalInfo {
                semaphore: self.0.semaphore,
                value,
                ..Default::default()
            })
        }
    }
    pub fn wait(
        self: Arc<TimelineSemaphore>,
        value: u64,
    ) -> impl Future<Output = VkResult<()>> + Send + Sync {
        blocking::unblock(move || {
            let value_this = value;
            let semaphore_this = self.0.semaphore;
            unsafe {
                self.0.device.wait_semaphores(
                    &vk::SemaphoreWaitInfo {
                        semaphore_count: 1,
                        p_semaphores: &semaphore_this,
                        p_values: &value_this,
                        ..Default::default()
                    },
                    std::u64::MAX,
                )?;
            }
            drop(self);
            Ok(())
        })
    }

    pub fn wait_n<const N: usize>(
        semaphores: [(Arc<TimelineSemaphore>, u64); N],
    ) -> impl Future<Output = VkResult<()>> {
        // Ensure that all semaphores have the same device
        let device = semaphores[0].0 .0.device.clone();
        for (semaphore, _) in semaphores.iter() {
            assert_eq!(semaphore.0.device.handle(), device.handle());
        }

        blocking::unblock(move || {
            let raw_semaphores: [vk::Semaphore; N] =
                semaphores.each_ref().map(|s| s.0 .0.semaphore);
            let nums: [u64; N] = semaphores.each_ref().map(|s| s.1);

            unsafe {
                device.wait_semaphores(
                    &vk::SemaphoreWaitInfo {
                        semaphore_count: N.try_into().unwrap(),
                        p_semaphores: raw_semaphores.as_ptr(),
                        p_values: nums.as_ptr(),
                        ..Default::default()
                    },
                    std::u64::MAX,
                )?;
            }
            drop(semaphores);
            drop(device);
            Ok(())
        })
    }

    pub fn wait_many(
        semaphores: Vec<TimelineStagedSemaphoreOp>,
    ) -> impl Future<Output = VkResult<()>> {
        // Ensure that all semaphores have the same device
        let device = semaphores[0].semaphore.0.device.clone();
        for op in semaphores.iter() {
            assert_eq!(op.semaphore.0.device.handle(), device.handle());
        }

        blocking::unblock(move || {
            let (raw_semaphores, nums): (Vec<vk::Semaphore>, Vec<u64>) = semaphores
                .iter()
                .map(|op| (op.semaphore.0.semaphore, op.value))
                .unzip();

            unsafe {
                device.wait_semaphores(
                    &vk::SemaphoreWaitInfo {
                        semaphore_count: raw_semaphores.len() as u32,
                        p_semaphores: raw_semaphores.as_ptr(),
                        p_values: nums.as_ptr(),
                        ..Default::default()
                    },
                    std::u64::MAX,
                )?;
            }
            drop(semaphores);
            drop(device);
            Ok(())
        })
    }

    /// Downgrade an Arc<TimelineSemaphore> into an Arc<Semaphore>.
    pub fn downgrade_arc(self: Arc<TimelineSemaphore>) -> Arc<Semaphore> {
        unsafe {
            // Safety: This relies on TimelineSemaphore being transmutable to Semaphore.
            std::mem::transmute(self)
        }
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.0.device
    }
}

pub struct TimelineStagedSemaphoreOp {
    pub semaphore: Arc<TimelineSemaphore>,
    pub value: u64,
}

impl TimelineStagedSemaphoreOp {
    pub fn wait(self) -> impl Future<Output = VkResult<()>> + Unpin {
        self.semaphore.wait(self.value)
    }
}
