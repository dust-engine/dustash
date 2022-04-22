use std::{ops::Deref, sync::Arc};

use thread_local::ThreadLocal;

use crate::Device;

use super::pool::CommandPool;
use ash::vk;

pub struct SharedCommandPool {
    pub(crate) device: Arc<Device>,
    pub(crate) pool: ThreadLocal<Arc<CommandPool>>,
    pub(crate) queue_family_index: u32,
}

impl Deref for SharedCommandPool {
    type Target = Arc<CommandPool>;

    fn deref(&self) -> &Self::Target {
        self.pool.get_or(|| {
            let command_pool = CommandPool::new(
                self.device.clone(),
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                self.queue_family_index,
            )
            .unwrap();
            Arc::new(command_pool)
        })
    }
}
