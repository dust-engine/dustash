use ash::vk;
use cstr::cstr;
use std::ffi::CStr;
use std::sync::Arc;

fn main() {
    let entry = unsafe { ash::Entry::load().unwrap() };
    let entry = Arc::new(entry);

    let instance = dustash::Instance::create(
        entry,
        &vk::InstanceCreateInfo::builder()
            .application_info(
                &vk::ApplicationInfo::builder()
                    .application_name(cstr!("Dustash Example"))
                    .application_version(vk::make_api_version(0, 0, 1, 0))
                    .api_version(vk::make_api_version(0, 1, 3, 0))
                    .build(),
            )
            .build(),
    )
    .unwrap();
    let instance = Arc::new(instance);

    let physical_devices = dustash::PhysicalDevice::enumerate(&instance).unwrap();
    let (device, queues) = physical_devices[0]
        .create_device(&[], &[], &vk::PhysicalDeviceFeatures::default())
        .unwrap();
}
