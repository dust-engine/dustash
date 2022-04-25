use std::{
    future::{Future, IntoFuture},
    mem::MaybeUninit,
    sync::Arc,
};

use ash::{prelude::VkResult, vk};

use crate::Device;

pub struct Fence {
    device: Arc<Device>,
    pub(crate) fence: vk::Fence,
}
impl crate::HasDevice for Fence {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl std::fmt::Debug for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Fence({:?})", self.fence))
    }
}
impl Fence {
    pub fn new(device: Arc<Device>, signaled: bool) -> VkResult<Self> {
        let mut flags = vk::FenceCreateFlags::empty();
        if signaled {
            flags |= vk::FenceCreateFlags::SIGNALED;
        }
        let fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::builder().flags(flags).build(), None)?
        };
        Ok(Self { device, fence })
    }
    pub fn signaled(&self) -> VkResult<bool> {
        unsafe { self.device.get_fence_status(self.fence) }
    }
    /// Blocks until the fence was signaled.
    pub fn wait(&self) -> VkResult<()> {
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, std::u64::MAX)
        }
    }
    pub fn join_n<const N: usize>(fences: [Fence; N]) -> FenceJoinN<N> {
        let device = fences[0].device.clone();
        let raw_fences: [vk::Fence; N] = fences.map(|f| {
            let fence = f.fence;
            let mut f = MaybeUninit::new(f); // This prevents dropping the fence.
            let device = unsafe {
                let mut uninit_arc: MaybeUninit<Arc<Device>> = MaybeUninit::uninit();
                std::ptr::swap(
                    &mut f.assume_init_mut().device as *mut Arc<Device>,
                    uninit_arc.as_mut_ptr(),
                );
                uninit_arc.assume_init()
            };
            // Manually drop device here
            drop(f);
            drop(device);
            fence
        });
        FenceJoinN {
            device,
            fences: raw_fences,
        }
    }
    pub fn join(fences: Vec<Fence>) -> FenceJoin {
        let device = fences[0].device.clone();
        let raw_fences: Vec<vk::Fence> = fences
            .into_iter()
            .map(|f| {
                let fence = f.fence;
                let mut f = MaybeUninit::new(f); // This prevents dropping the fence.
                let device = unsafe {
                    let mut uninit_arc: MaybeUninit<Arc<Device>> = MaybeUninit::uninit();
                    std::ptr::swap(
                        &mut f.assume_init_mut().device as *mut Arc<Device>,
                        uninit_arc.as_mut_ptr(),
                    );
                    uninit_arc.assume_init()
                };
                // Manually drop device here
                drop(f);
                drop(device);
                fence
            })
            .collect();
        FenceJoin {
            device,
            fences: raw_fences,
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        tracing::debug!(fence = ?self.fence, "drop fence");
        // Safety: Host Syncronization rule for vkDestroyFence:
        // - Host access to fence must be externally synchronized
        // We have &mut self and thus exclusive control on the fence.
        unsafe {
            self.device.destroy_fence(self.fence, None);
        }
    }
}

impl IntoFuture for Fence {
    type Output = VkResult<()>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        blocking::unblock(move || self.wait())
    }
}

pub struct FenceJoinN<const N: usize> {
    device: Arc<Device>,
    fences: [vk::Fence; N],
}

impl<const N: usize> FenceJoinN<N> {
    /// Blocks until all fences are signaled
    pub fn wait(&self) -> VkResult<()> {
        unsafe {
            self.device
                .wait_for_fences(self.fences.as_slice(), true, std::u64::MAX)
        }
    }
}

impl<const N: usize> Drop for FenceJoinN<N> {
    fn drop(&mut self) {
        // Safety: Host Syncronization rule for vkDestroyFence:
        // - Host access to fence must be externally synchronized
        // We have &mut self and thus exclusive control on the fence.
        unsafe {
            for fence in self.fences.iter() {
                self.device.destroy_fence(*fence, None);
            }
        }
    }
}

impl<const N: usize> IntoFuture for FenceJoinN<N> {
    type Output = VkResult<()>;

    type IntoFuture = impl Future<Output = VkResult<()>>;

    fn into_future(self) -> Self::IntoFuture {
        blocking::unblock(move || self.wait())
    }
}

pub struct FenceJoin {
    device: Arc<Device>,
    fences: Vec<vk::Fence>,
}

impl FenceJoin {
    /// Blocks until all fences are signaled
    pub fn wait(&self) -> VkResult<()> {
        unsafe {
            self.device
                .wait_for_fences(self.fences.as_slice(), true, std::u64::MAX)
        }
    }
}

impl Drop for FenceJoin {
    fn drop(&mut self) {
        // Safety: Host Syncronization rule for vkDestroyFence:
        // - Host access to fence must be externally synchronized
        // We have &mut self and thus exclusive control on the fence.
        unsafe {
            for fence in self.fences.iter() {
                tracing::debug!(fence = ?fence, "drop fences");
                self.device.destroy_fence(*fence, None);
            }
        }
    }
}

impl IntoFuture for FenceJoin {
    type Output = VkResult<()>;

    type IntoFuture = impl Future<Output = VkResult<()>>;

    fn into_future(self) -> Self::IntoFuture {
        blocking::unblock(move || self.wait())
    }
}
