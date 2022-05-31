#![feature(into_future)]
#![feature(type_alias_impl_trait)]
#![feature(negative_impls)]
#![feature(array_methods)]
#![feature(maybe_uninit_uninit_array)]
#![feature(const_for)]
#![feature(const_option_ext)]
#![feature(alloc_layout_extra)]
#![feature(int_roundings)]
#![feature(core_ffi_c)]
#![feature(iterator_try_collect)]
#![feature(array_zip)]

use std::ops::Deref;

pub mod command;
mod debug;
pub use debug::DebugUtilsMessenger;
pub mod accel_struct;
pub mod fence;
pub mod frames;
mod physical_device;
pub mod queue;
pub mod resources;
pub mod surface;
pub mod swapchain;
pub use physical_device::*;
pub mod descriptor;
pub mod ray_tracing;
pub mod shader;
pub mod sync;

mod device;
mod instance;

pub use blocking::Task;
pub use debug::DebugObject;
pub use device::{Device, HasDevice};
pub use instance::Instance;
