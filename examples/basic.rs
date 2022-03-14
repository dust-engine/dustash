use ash::extensions::khr;
use ash::vk;
use cstr::cstr;
use dustash::frames::FrameManager;
use raw_window_handle::HasRawWindowHandle;
use std::panic::UnwindSafe;
use std::sync::Arc;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop},
    window::WindowBuilder,
};
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
            .enabled_extension_names(&[
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(),
            ])
            .build(),
    )
    .unwrap();
    let instance = Arc::new(instance);

    let physical_devices = dustash::PhysicalDevice::enumerate(&instance).unwrap();
    let (device, queues) = physical_devices[0]
        .clone()
        .create_device(
            &[],
            &[khr::Swapchain::name()],
            &vk::PhysicalDeviceFeatures::default(),
        )
        .unwrap();

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let surface_loader = Arc::new(dustash::surface::SurfaceLoader::new(instance.clone()));
    let swapchain_loader = Arc::new(dustash::swapchain::SwapchainLoader::new(device.clone()));
    let surface = dustash::surface::Surface::create(surface_loader, &window).unwrap();
    let surface = Arc::new(surface);
    let mut frames = FrameManager::new(
        swapchain_loader,
        surface,
        dustash::frames::Options::default(),
        vk::Extent2D {
            width: 1280,
            height: 720,
        },
    )
    .unwrap();
    window_update(window, event_loop, move || {
        let frame = frames.acquire(!0).unwrap();
        println!("HELLO");
    });
}

fn window_update(
    window: winit::window::Window,
    event_loop: EventLoop<()>,
    mut update_fn: impl FnMut() -> () + 'static,
) {
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => update_fn(),
            _ => (),
        }
    });
}
