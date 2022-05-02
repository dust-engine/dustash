use std::{future::Future, sync::Arc};

use ash::{prelude::VkResult, vk};

use crate::Device;

use super::SemaphoreOp;

pub struct Semaphore {
    device: Arc<Device>,
    pub(crate) semaphore: vk::Semaphore,
}

impl crate::HasDevice for Semaphore {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
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
    pub unsafe fn as_timeline_arc(self: Arc<Self>) -> Arc<TimelineSemaphore> {
        std::mem::transmute(self)
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

impl std::fmt::Debug for TimelineSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("TimelineSemaphore({:?})", self.0.semaphore))
    }
}

impl crate::HasDevice for TimelineSemaphore {
    fn device(&self) -> &Arc<Device> {
        self.0.device()
    }
}

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
    pub fn value(&self) -> VkResult<u64> {
        unsafe { self.0.device.get_semaphore_counter_value(self.0.semaphore) }
    }
    pub fn block(self: &TimelineSemaphore, value: u64) -> VkResult<()> {
        unsafe {
            self.0.device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: 1,
                    p_semaphores: &self.0.semaphore,
                    p_values: &value,
                    ..Default::default()
                },
                std::u64::MAX,
            )
        }
    }
    pub fn wait(
        self: Arc<TimelineSemaphore>,
        value: u64,
    ) -> impl Future<Output = VkResult<()>> + Send + Sync {
        blocking::unblock(move || {
            self.block(value)?;
            drop(self);
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
}

#[derive(Clone)]
pub struct TimelineSemaphoreOp {
    pub semaphore: Arc<TimelineSemaphore>,
    pub value: u64,
}

impl std::fmt::Debug for TimelineSemaphoreOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "TimelineSemaphore({:?})[{}]",
            self.semaphore.0.semaphore, self.value
        ))
    }
}

impl TimelineSemaphoreOp {
    pub fn block(self) -> VkResult<()> {
        self.semaphore.block(self.value)
    }
    pub fn wait(self) -> impl Future<Output = VkResult<()>> + Unpin {
        self.semaphore.wait(self.value)
    }

    pub fn block_n<const N: usize>(semaphores: [&TimelineSemaphoreOp; N]) -> VkResult<()> {
        let device = semaphores[0].semaphore.0.device.clone();
        // Ensure that all semaphores have the same device
        for op in semaphores.iter().skip(1) {
            assert_eq!(op.semaphore.0.device.handle(), device.handle());
        }
        let semaphore_values = semaphores.map(|s| s.semaphore.0.semaphore);
        let values = semaphores.map(|s| s.value);
        unsafe {
            device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: N as u32,
                    p_semaphores: semaphore_values.as_ptr(),
                    p_values: values.as_ptr(),
                    ..Default::default()
                },
                std::u64::MAX,
            )
        }
    }
    pub fn wait_n<const N: usize>(
        semaphores: [TimelineSemaphoreOp; N],
    ) -> impl Future<Output = VkResult<()>> {
        blocking::unblock(move || {
            Self::block_n(semaphores.each_ref())?;
            drop(semaphores);
            Ok(())
        })
    }
    pub fn block_many(semaphores: &[&TimelineSemaphoreOp]) -> VkResult<()> {
        if semaphores.len() == 0 {
            return Ok(());
        }
        let device = semaphores[0].semaphore.0.device.clone();
        // Ensure that all semaphores have the same device
        for op in semaphores.iter().skip(1) {
            assert_eq!(op.semaphore.0.device.handle(), device.handle());
        }
        let semaphore_values: Vec<_> = semaphores.iter().map(|s| s.semaphore.0.semaphore).collect();
        let values: Vec<_> = semaphores.iter().map(|s| s.value).collect();
        unsafe {
            device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: semaphore_values.len() as u32,
                    p_semaphores: semaphore_values.as_ptr(),
                    p_values: values.as_ptr(),
                    ..Default::default()
                },
                std::u64::MAX,
            )
        }
    }

    pub fn wait_many(semaphores: Vec<TimelineSemaphoreOp>) -> impl Future<Output = VkResult<()>> {
        // Ensure that all semaphores have the same device
        blocking::unblock(move || {
            let refs: Vec<_> = semaphores.iter().collect();
            Self::block_many(&refs)?;
            drop(semaphores);
            Ok(())
        })
    }
    pub fn signal(&self) -> VkResult<()> {
        self.semaphore.signal(self.value)
    }
    pub fn finished(&self) -> VkResult<bool> {
        let val = self.semaphore.value()?;
        Ok(val >= self.value)
    }
    pub fn downgrade_arc(self) -> SemaphoreOp {
        SemaphoreOp {
            value: self.value,
            semaphore: self.semaphore.downgrade_arc(),
        }
    }
    pub fn increment(self) -> Self {
        Self {
            semaphore: self.semaphore,
            value: self.value + 1,
        }
    }
}
