use ash::extensions::{khr, ext};
use ash::vk;
use cstr::cstr;
use dustash::DebugUtilsMessenger;
use dustash::queue::QueueType;
use dustash::{command::pool::CommandPool, frames::FrameManager};

use std::sync::Arc;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
fn main() {
    tracing_subscriber::fmt::init();
    let entry = unsafe { ash::Entry::load().unwrap() };
    let entry = Arc::new(entry);
    let version = entry.try_enumerate_instance_version().unwrap().unwrap();
    println!(
        "Supported version: {}.{}.{}({})",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version),
        vk::api_version_variant(version)
    );

    let mut instance = dustash::Instance::create(
        entry.clone(),
        &vk::InstanceCreateInfo::builder()
            .application_info(
                &vk::ApplicationInfo::builder()
                    .application_name(cstr!("Dustash Example"))
                    .application_version(vk::make_api_version(0, 0, 1, 0))
                    .api_version(version)
                    .build(),
            )
            .enabled_extension_names(&[
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(),
                ext::DebugUtils::name().as_ptr(),
            ])
            .build(),
    )
    .unwrap();
    let instance = Arc::new(instance);

    let physical_devices = dustash::PhysicalDevice::enumerate(&instance).unwrap();
    let (device, mut queues) = physical_devices[0]
        .clone()
        .create_device(
            &[],
            &[khr::Swapchain::name()],
            &vk::PhysicalDeviceFeatures2::builder().push_next(
                &mut vk::PhysicalDeviceSynchronization2Features {
                    synchronization2: vk::TRUE,
                    ..Default::default()
                },
            ),
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
        dustash::frames::Options {
            usage: vk::ImageUsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        vk::Extent2D {
            width: 1280,
            height: 720,
        },
    )
    .unwrap();

    let command_pool = CommandPool::new(
        device.clone(),
        vk::CommandPoolCreateFlags::empty(),
        queues.of_type(QueueType::Compute).family_index(),
    )
    .unwrap();
    let command_pool = Arc::new(command_pool);
    window_update(window, event_loop, move || {
        let acquired_image = frames.acquire(!0).unwrap();
        let buffer = command_pool.clone().allocate_one().unwrap();
        let exec = buffer
            .record(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, |cr| {
                let clear_color_value = vk::ClearColorValue {
                    float32: [0.0, 1.0, 0.0, 1.0],
                };
                cr.pipeline_barrier(
                    &vk::DependencyInfo::builder()
                        .dependency_flags(vk::DependencyFlags::BY_REGION)
                        .image_memory_barriers(&[vk::ImageMemoryBarrier2 {
                            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                            dst_stage_mask: vk::PipelineStageFlags2::CLEAR,
                            src_access_mask: vk::AccessFlags2::NONE,
                            dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::GENERAL,
                            image: acquired_image.image,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_array_layer: 0,
                                layer_count: 1,
                                base_mip_level: 0,
                                level_count: 1,
                            },
                            ..Default::default()
                        }])
                        .build(),
                )
                .clear_color_image(
                    acquired_image.image,
                    vk::ImageLayout::GENERAL,
                    &clear_color_value,
                    &[vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        base_mip_level: 0,
                        level_count: 1,
                    }],
                )
                .pipeline_barrier(
                    &vk::DependencyInfo::builder()
                        .dependency_flags(vk::DependencyFlags::BY_REGION)
                        .image_memory_barriers(&[vk::ImageMemoryBarrier2 {
                            src_stage_mask: vk::PipelineStageFlags2::CLEAR,
                            dst_stage_mask: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                            dst_access_mask: vk::AccessFlags2::NONE,
                            old_layout: vk::ImageLayout::GENERAL,
                            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                            image: acquired_image.image,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_array_layer: 0,
                                layer_count: 1,
                                base_mip_level: 0,
                                level_count: 1,
                            },
                            ..Default::default()
                        }])
                        .build(),
                );
            })
            .unwrap();

        queues
            .of_type(QueueType::Compute)
            .submit(
                vec![dustash::queue::SemaphoreOp::binary(
                    acquired_image.ready_semaphore.clone(),
                    vk::PipelineStageFlags2::CLEAR,
                )],
                vec![exec],
                vec![dustash::queue::SemaphoreOp::binary(
                    acquired_image.complete_semaphore.clone(),
                    vk::PipelineStageFlags2::CLEAR,
                )],
            )
            .fence(acquired_image.complete_fence.clone());
        queues
            .flush_and_present(&mut frames, acquired_image)
            .unwrap();
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
