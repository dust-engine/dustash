//! Original algorithm obtained from https://github.com/Tobski/simple_vulkan_synchronization
//! # Simpler Vulkan Synchronization
//! In an effort to make Vulkan synchronization more accessible, I created this
//! stb-inspired single-header library in order to somewhat simplify the core
//! synchronization mechanisms in Vulkan - pipeline barriers and events.
//!
//! Rather than the complex maze of enums and bit flags in Vulkan - many
//! combinations of which are invalid or nonsensical - this library collapses
//! this to a much shorter list of 40 distinct usage types, and a couple of
//! options for handling image layouts.
//!
//! Use of other synchronization mechanisms such as semaphores, fences and render
//! passes are not addressed in this API at present.
//!
//! ## EXPRESSIVENESS COMPARED TO RAW VULKAN
//!
//! Despite the fact that this API is fairly simple, it expresses 99% of
//! what you'd actually ever want to do in practice.
//! Adding the missing expressiveness would result in increased complexity
//! which didn't seem worth the trade off - however I would consider adding
//! something for them in future if it becomes an issue.
//!
//! Here's a list of known things you can't express:
//! - Execution only dependencies cannot be expressed.
//! These are occasionally useful in conjunction with semaphores, or when
//! trying to be clever with scheduling - but their usage is both limited
//! and fairly tricky to get right anyway.
//! - Depth/Stencil Input Attachments can be read in a shader using either
//! VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL or
//! VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL - this library
//! *always* uses VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL.
//! It's possible (though highly unlikely) when aliasing images that this
//! results in unnecessary transitions.
//!
//!
//! Testing of this library is so far extremely limited with no immediate
//! plans to add to that - so there's bound to be some amount of bugs.
//! Please raise these issues on the repo issue tracker, or provide a fix
//! via a pull request yourself if you're so inclined.

use std::{mem::MaybeUninit, ptr::null};

use super::recorder::CommandRecorder;
use ash::vk;

#[derive(Copy, Clone)]
pub enum AccessType {
    None,
    /// Command buffer read operation as defined by NV_device_generated_commands.
    /// Requires VK_NV_device_generated_commands to be enabled.
    CommandBufferReadNV,
    /// Read as an indirect buffer for drawing or dispatch
    IndirectBuffer,
    /// Read as an index buffer for drawing
    IndexBuffer,
    /// Read as a vertex buffer for drawing
    VertexBuffer,
    /// Read as a uniform buffer in a vertex shader
    VertexShaderReadUniformBuffer,
    /// Read as a sampled image/uniform texel buffer in a vertex shader
    VertexShaderReadSampledImageOrUniformTexelBuffer,
    /// Read as any other resource in a vertex shader
    VertexShaderReadOther,
    /// Read as a uniform buffer in a tessellation control shader
    TessellationControlShaderReadUniformBuffer,
    /// Read as a sampled image/uniform texel buffer  in a tessellation control shader
    TessellationControlShaderReadSampledImageOrUniformTexelBuffer,
    /// Read as any other resource in a tessellation control shader
    TessellationControlShaderReadOther,
    /// Read as a uniform buffer in a tessellation evaluation shader
    TessellationEvaluationShaderReadUniformBuffer,
    /// Read as a sampled image/uniform texel buffer in a tessellation evaluation shader
    TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer,
    /// Read as any other resource in a tessellation evaluation shader
    TessellationEvaluationShaderReadOther,
    /// Read as a uniform buffer in a geometry shader
    GeometryShaderReadUniformBuffer,
    /// Read as a sampled image/uniform texel buffer  in a geometry shader
    GeometryShaderReadSampledImageOrUniformTexelBuffer,
    /// Read as any other resource in a geometry shader
    GeometryShaderReadOther,
    /// Read as a uniform buffer in a task shader
    TaskShaderReadUniformBufferNV,
    /// Read as a sampled image/uniform texel buffer in a task shader
    TaskShaderReadSampledImageOrUniformTexelBufferNV,
    /// Read as any other resource in a task shader
    TaskShaderReadOtherNV,
    /// Read as a uniform buffer in a mesh shader
    MeshShaderReadUniformBufferNV,
    /// Read as a sampled image/uniform texel buffer in a mesh shader
    MeshShaderReadSampledImageOrUniformTexelBufferNV,
    /// Read as any other resource in a mesh shader
    MeshShaderReadOtherNV,
    /// Read as a transform feedback counter buffer
    TransformFeedbackCounterReadEXT,
    /// Read as a fragment density map image
    FragmentDensityMapReadEXT,
    /// Read as a shading rate image
    ShadingRateReadNV,
    /// Read as a uniform buffer in a fragment shader
    FragmentShaderReadUniformBuffer,
    /// Read as a sampled image/uniform texel buffer  in a fragment shader
    FragmentShaderReadSampledImageOrUniformTexelBuffer,
    /// Read as an input attachment with a color format in a fragment shader
    FragmentShaderReadColorInputAttachment,
    /// Read as an input attachment with a depth/stencil format in a fragment shader
    FragmentShaderReadDepthStencilInputAttachment,
    /// Read as any other resource in a fragment shader
    FragmentShaderReadOther,
    /// Read by standard blending/logic operations or subpass load operations
    ColorAttachmentRead,
    /// Read by advanced blending, standard blending, logic operations, or subpass load operations
    ColorAttachmentAdvancedBlendingEXT,
    /// Read by depth/stencil tests or subpass load operations
    DepthStencilAttachmentRead,
    /// Read as a uniform buffer in a compute shader
    ComputeShaderReadUniformBuffer,
    /// Read as a sampled image/uniform texel buffer in a compute shader
    ComputeShaderReadSampledImageOrUniformTexelBuffer,
    /// Read as any other resource in a compute shader
    ComputeShaderReadOther,
    /// Read as a uniform buffer in any shader
    AnyShaderReadUniformBuffer,
    /// Read as a uniform buffer in any shader, or a vertex buffer
    AnyShaderReadUniformBufferOrVertexBuffer,
    /// Read as a sampled image in any shader
    AnyShaderReadSampledImageOrUniformTexelBuffer,
    /// Read as any other resource (excluding attachments) in any shader
    AnyShaderReadOther,
    /// Read as the source of a transfer operation
    TransferRead,
    /// Read on the host
    HostRead,

    /// Read by conditional rendering.
    /// Requires VK_EXT_conditional_rendering to be enabled.
    ConditionalRenderingReadEXT,

    /// Read by a ray tracing shader as an acceleration structure.
    /// Requires VK_KHR_ray_tracing to be enabled.
    RayTracingShaderAccelerationStructureReadKHR,
    /// Read as an acceleration structure during a build
    /// Requires VK_KHR_acceleration_structure to be enabled.
    AccelerationStructureBuildReadKHR,

    /// Read by the presentation engine (i.e. vkQueuePresentKHR).
    /// Requires VK_KHR_swapchain to be enabled.
    Present,
    // Note that Present should always be the last Read access.
    // The implementation of is_read() and is_write() depends on this.

    // Write access
    /// Command buffer write operation.
    /// Requires VK_NV_device_generated_commands to be enabled/
    CommandBufferWriteNV,
    /// Written as any resource in a vertex shader
    VertexShaderWrite,
    /// Written as any resource in a tessellation control shader
    TessellationControlShaderWrite,
    /// Written as any resource in a tessellation evaluation shader
    TessellationEvaluationShaderWrite,
    /// Written as any resource in a geometry shader
    GeometryShaderWrite,

    /// Written as any resource in a task shader.
    /// Requires VK_NV_mesh_shading to be enabled.
    TaskShaderWriteNV,
    /// Written as any resource in a mesh shader
    MeshShaderWriteNV,

    // Requires VK_EXT_transform_feedback to be enabled
    /// Written as a transform feedback buffer
    TransformFeedbackWriteEXT,
    /// Written as a transform feedback counter buffer
    TransformFeedbackCounterWriteEXT,

    /// Written as any resource in a fragment shader
    FragmentShaderWrite,
    /// Written as a color attachment during rendering, or via a subpass store op
    ColorAttachmentWrite,
    /// Written as a depth/stencil attachment during rendering, or via a subpass store op
    DepthStencilAttachmentWrite,

    // Requires VK_KHR_maintenance2 to be enabled
    /// Written as a depth aspect of a depth/stencil attachment during rendering, whilst the stencil aspect is read-only
    DepthAttachmentWriteStencilReadOnly,
    /// Written as a stencil aspect of a depth/stencil attachment during rendering, whilst the depth aspect is read-only
    StencilAttachmentWriteDepthReadOnly,

    /// Written as any resource in a compute shader
    ComputeShaderWrite,
    /// Written as any resource in any shader
    AnyShaderWrite,
    /// Written as the destination of a transfer operation
    TransferWrite,
    /// Data pre-filled by host before device access starts
    HostPreinitialized,
    /// Written on the host
    HostWrite,

    /// Written as an acceleration structure during a build.
    /// Requires VK_KHR_ray_tracing to be enabled.
    AccelerationStructureBuildWriteKHR,

    /// Read or written as a color attachment during rendering
    ColorAttachmentReadWrite,
    // General access
    /// Covers any access - useful for debug, generally avoid for performance reasons
    General,
}

struct VkAccessInfo {
    stage_mask: vk::PipelineStageFlags,
    access_mask: vk::AccessFlags,
    image_layout: vk::ImageLayout,
}

impl AccessType {
    const fn is_read_only(&self) -> bool {
        let this = *self as u32;
        this <= (Self::Present as u32)
    }
    const fn is_write(&self) -> bool {
        (*self as u32) > (Self::Present as u32)
    }
    const fn to_vk(&self) -> VkAccessInfo {
        match self {
            AccessType::None => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::empty(),
                access_mask: vk::AccessFlags::empty(),
                image_layout: vk::ImageLayout::UNDEFINED
            },
            AccessType::CommandBufferReadNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COMMAND_PREPROCESS_NV,
                access_mask: vk::AccessFlags::COMMAND_PREPROCESS_READ_NV,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::IndirectBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::DRAW_INDIRECT,
                access_mask: vk::AccessFlags::INDIRECT_COMMAND_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::IndexBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::VERTEX_INPUT,
                access_mask: vk::AccessFlags::INDEX_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::VERTEX_INPUT,
                access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::VertexShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationControlShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer => {
                VkAccessInfo {
                    stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                    access_mask: vk::AccessFlags::SHADER_READ,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }
            }
            AccessType::TessellationControlShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationEvaluationShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer => {
                VkAccessInfo {
                    stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                    access_mask: vk::AccessFlags::SHADER_READ,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }
            }
            AccessType::TessellationEvaluationShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::GeometryShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TaskShaderReadUniformBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TASK_SHADER_NV,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TaskShaderReadSampledImageOrUniformTexelBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TASK_SHADER_NV,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::TaskShaderReadOtherNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TASK_SHADER_NV,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::MeshShaderReadUniformBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::MESH_SHADER_NV,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::MeshShaderReadSampledImageOrUniformTexelBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::MESH_SHADER_NV,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::MeshShaderReadOtherNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::MESH_SHADER_NV,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransformFeedbackCounterReadEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentDensityMapReadEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::FRAGMENT_DENSITY_PROCESS_EXT,
                access_mask: vk::AccessFlags::FRAGMENT_DENSITY_MAP_READ_EXT,
                image_layout: vk::ImageLayout::FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
            },
            AccessType::ShadingRateReadNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::SHADING_RATE_IMAGE_NV,
                access_mask: vk::AccessFlags::SHADING_RATE_IMAGE_READ_NV,
                image_layout: vk::ImageLayout::SHADING_RATE_OPTIMAL_NV,
            },
            AccessType::FragmentShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadColorInputAttachment => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadDepthStencilInputAttachment => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ColorAttachmentRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::ColorAttachmentAdvancedBlendingEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::from_raw(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            AccessType::ComputeShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::ComputeShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                access_mask: vk::AccessFlags::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AnyShaderReadUniformBufferOrVertexBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                access_mask: vk::AccessFlags::from_raw(
                    vk::AccessFlags::UNIFORM_READ.as_raw()
                        | vk::AccessFlags::VERTEX_ATTRIBUTE_READ.as_raw(),
                ),
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::AnyShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransferRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TRANSFER,
                access_mask: vk::AccessFlags::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::HostRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::HOST,
                access_mask: vk::AccessFlags::HOST_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::Present => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::empty(),
                access_mask: vk::AccessFlags::empty(),
                image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
            AccessType::ConditionalRenderingReadEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT,
                access_mask: vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::RayTracingShaderAccelerationStructureReadKHR => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AccelerationStructureBuildReadKHR => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::CommandBufferWriteNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COMMAND_PREPROCESS_NV,
                access_mask: vk::AccessFlags::COMMAND_PREPROCESS_WRITE_NV,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationControlShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationEvaluationShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TaskShaderWriteNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TASK_SHADER_NV,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::MeshShaderWriteNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::MESH_SHADER_NV,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransformFeedbackWriteEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags::TRANSFORM_FEEDBACK_WRITE_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TransformFeedbackCounterWriteEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ColorAttachmentWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::from_raw(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthAttachmentWriteStencilReadOnly => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::from_raw(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags::from_raw(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ.as_raw(),
                ),
                image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
            },
            AccessType::StencilAttachmentWriteDepthReadOnly => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::from_raw(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags::from_raw(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ.as_raw(),
                ),
                image_layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
            },
            AccessType::ComputeShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                access_mask: vk::AccessFlags::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransferWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::TRANSFER,
                access_mask: vk::AccessFlags::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::HostPreinitialized => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::HOST,
                access_mask: vk::AccessFlags::HOST_WRITE,
                image_layout: vk::ImageLayout::PREINITIALIZED,
            },
            AccessType::HostWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::HOST,
                access_mask: vk::AccessFlags::HOST_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AccelerationStructureBuildWriteKHR => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::ColorAttachmentReadWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags::from_raw(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ.as_raw()
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw(),
                ),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::General => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                access_mask: vk::AccessFlags::from_raw(
                    vk::AccessFlags::MEMORY_READ.as_raw() | vk::AccessFlags::MEMORY_WRITE.as_raw(),
                ),
                image_layout: vk::ImageLayout::GENERAL,
            },
        }
    }
}

/// ImageLayout defines a handful of layout options for images.
/// Rather than a list of all possible image layouts, this reduced list is
/// correlated with the access types to map to the correct Vulkan layouts.
/// ImageLayout::Optimal is usually preferred.
pub enum ImageLayout {
    /// Choose the most optimal layout for each usage. Performs layout transitions as appropriate for the access.
    Optimal,
    /// Layout accessible by all Vulkan access types on a device - no layout transitions except for presentation.
    General,

    /// As ImageLayout::General, but also allows presentation engines to access it - no layout transitions.
    /// Requires VK_KHR_shared_presentable_image to be enabled. Can only be used for shared presentable images (i.e. single-buffered swap chains).
    GeneralAndPresentation,
}

/// Global barriers define a set of accesses on multiple resources at once.
/// If a buffer or image doesn't require a queue ownership transfer, or an image
/// doesn't require a layout transition (e.g. you're using one of the GENERAL
/// layouts) then a global barrier should be preferred.
/// Simply define the previous and next access types of resources affected.
pub struct MemoryBarrier<'a> {
    pub prev_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
}

/// Buffer barriers should only be used when a queue family ownership transfer
/// is required - prefer global barriers at all other times.
///
/// Access types are defined in the same way as for a global memory barrier, but
/// they only affect the buffer range identified by buffer, offset and size,
/// rather than all resources.
///
/// srcQueueFamilyIndex and dstQueueFamilyIndex will be passed unmodified into a
/// VkBufferMemoryBarrier.
///
/// A buffer barrier defining a queue ownership transfer needs to be executed
/// twice - once by a queue in the source queue family, and then once again by a
/// queue in the destination queue family, with a semaphore guaranteeing
/// execution order between them.
pub struct BufferBarrier<'a> {
    pub memory_barrier: MemoryBarrier<'a>,

    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

/// Image barriers should only be used when a queue family ownership transfer
/// or an image layout transition is required - prefer global barriers at all
/// other times.
/// In general it is better to use image barriers with THSVS_IMAGE_LAYOUT_OPTIMAL
/// than it is to use global barriers with images using either of the
/// THSVS_IMAGE_LAYOUT_GENERAL* layouts.

/// Access types are defined in the same way as for a global memory barrier, but
/// they only affect the image subresource range identified by image and
/// subresourceRange, rather than all resources.
/// srcQueueFamilyIndex, dstQueueFamilyIndex, image, and subresourceRange will
/// be passed unmodified into a VkImageMemoryBarrier.

/// An image barrier defining a queue ownership transfer needs to be executed
/// twice - once by a queue in the source queue family, and then once again by a
/// queue in the destination queue family, with a semaphore guaranteeing
/// execution order between them.

/// If discard_contents is set to true, the contents of the image become
/// undefined after the barrier is executed, which can result in a performance
/// boost over attempting to preserve the contents.
/// This is particularly useful for transient images where the contents are
/// going to be immediately overwritten. A good example of when to use this is
/// when an application re-uses a presented image after vkAcquireNextImageKHR.
pub struct ImageBarrier<'a> {
    pub memory_barrier: MemoryBarrier<'a>,
    pub prev_layout: ImageLayout,
    pub next_layout: ImageLayout,
    pub discard_contents: bool,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: vk::Image,
    pub subresource_range: vk::ImageSubresourceRange,
}

/// Returns: (srcStages, dstStages)
impl<'a> MemoryBarrier<'a> {
    const fn to_vk(
        &self,
    ) -> (
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
        vk::MemoryBarrier,
    ) {
        let mut src_stages = vk::PipelineStageFlags::empty();
        let mut dst_stages = vk::PipelineStageFlags::empty();

        let mut barrier = vk::MemoryBarrier {
            s_type: vk::StructureType::MEMORY_BARRIER,
            p_next: null(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
        };
        let mut i: usize = 0;
        while i < self.prev_accesses.len() {
            let prev_access = &self.prev_accesses[i];
            let info = prev_access.to_vk();
            assert!(
                prev_access.is_read_only() || self.prev_accesses.len() == 1,
                "Multiple Writes"
            );
            src_stages =
                vk::PipelineStageFlags::from_raw(src_stages.as_raw() | info.stage_mask.as_raw());
            if prev_access.is_write() {
                barrier.src_access_mask = vk::AccessFlags::from_raw(
                    barrier.src_access_mask.as_raw() | info.access_mask.as_raw(),
                );
            }
            i = i + 1;
        }
        i = 0;
        while i < self.next_accesses.len() {
            let next_access = &self.next_accesses[i];
            let info = next_access.to_vk();
            assert!(
                next_access.is_read_only() || self.next_accesses.len() == 1,
                "Multiple Writes"
            );
            dst_stages =
                vk::PipelineStageFlags::from_raw(dst_stages.as_raw() | info.stage_mask.as_raw());
            if !barrier.src_access_mask.is_empty() {
                barrier.dst_access_mask = vk::AccessFlags::from_raw(
                    barrier.dst_access_mask.as_raw() | info.access_mask.as_raw(),
                );
            }
            i = i + 1;
        }
        if src_stages.is_empty() {
            src_stages = vk::PipelineStageFlags::TOP_OF_PIPE;
        }
        if dst_stages.is_empty() {
            dst_stages = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        }
        (src_stages, dst_stages, barrier)
    }
}

impl<'a> BufferBarrier<'a> {
    const fn to_vk(
        &self,
    ) -> (
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
        vk::BufferMemoryBarrier,
    ) {
        assert!(self.src_queue_family_index != self.dst_queue_family_index, "BufferBarrier should only be used when a queue family ownership transfer is required. Use MemoryBarrier instead.");
        let (src_stages, dst_stages, barrier) = self.memory_barrier.to_vk();
        let barrier = vk::BufferMemoryBarrier {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
            p_next: null(),
            src_access_mask: barrier.src_access_mask,
            dst_access_mask: barrier.dst_access_mask,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
        };
        (src_stages, dst_stages, barrier)
    }
}

impl<'a> ImageBarrier<'a> {
    const fn to_vk(
        &self,
    ) -> (
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
        vk::ImageMemoryBarrier,
    ) {
        let (src_stages, dst_stages, barrier) = self.memory_barrier.to_vk();
        let mut barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: null(),
            src_access_mask: barrier.src_access_mask,
            dst_access_mask: barrier.dst_access_mask,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            image: self.image,
            subresource_range: self.subresource_range,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::UNDEFINED,
        };

        if !self.discard_contents {
            let mut i = 0;
            while i < self.memory_barrier.prev_accesses.len() {
                let prev_access = self.memory_barrier.prev_accesses[i];
                let info = prev_access.to_vk();
                let layout = match self.prev_layout {
                    ImageLayout::General if prev_access as u32 == AccessType::Present as u32 => {
                        vk::ImageLayout::PRESENT_SRC_KHR
                    }
                    ImageLayout::General => vk::ImageLayout::GENERAL,
                    ImageLayout::Optimal => info.image_layout,
                    ImageLayout::GeneralAndPresentation => vk::ImageLayout::SHARED_PRESENT_KHR,
                };
                assert!(
                    barrier.old_layout.as_raw() == vk::ImageLayout::UNDEFINED.as_raw()
                        || barrier.old_layout.as_raw() == layout.as_raw(),
                    "Mixed Image Layout"
                );
                barrier.old_layout = layout;
                i = i + 1;
            }
        }

        {
            let mut i = 0;
            while i < self.memory_barrier.next_accesses.len() {
                let next_access = &self.memory_barrier.next_accesses[i];
                let info = next_access.to_vk();
                let layout = match self.next_layout {
                    ImageLayout::General if *next_access as u32 == AccessType::Present as u32 => {
                        vk::ImageLayout::PRESENT_SRC_KHR
                    }
                    ImageLayout::General => vk::ImageLayout::GENERAL,
                    ImageLayout::Optimal => info.image_layout,
                    ImageLayout::GeneralAndPresentation => vk::ImageLayout::SHARED_PRESENT_KHR,
                };
                assert!(
                    barrier.new_layout.as_raw() == vk::ImageLayout::UNDEFINED.as_raw()
                        || barrier.new_layout.as_raw() == layout.as_raw(),
                    "Mixed Image Layout"
                );
                barrier.new_layout = layout;
                i = i + 1;
            }
        }

        assert!(barrier.new_layout.as_raw() != barrier.old_layout.as_raw() || barrier.src_queue_family_index != barrier.dst_queue_family_index, "Image barriers should only be used when a queue family ownership transfer or an image layout transition is required. Use MemoryBarrier instead.");

        (src_stages, dst_stages, barrier)
    }
}

pub struct PipelineBarrierConst<const BL: usize, const IL: usize> {
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flag: vk::DependencyFlags,
    memory_barrier: Option<vk::MemoryBarrier>,
    buffer_barriers: [vk::BufferMemoryBarrier; BL],
    image_barriers: [vk::ImageMemoryBarrier; IL],
}

impl<const BL: usize, const IL: usize> PipelineBarrierConst<BL, IL> {
    pub const fn new(
        memory_barrier: Option<MemoryBarrier>,
        buffer_barriers: &[BufferBarrier; BL],
        image_barriers: &[ImageBarrier; IL],
        dependency_flag: vk::DependencyFlags,
    ) -> Self {
        let mut src_stage_mask: u32 = vk::PipelineStageFlags::TOP_OF_PIPE.as_raw();
        let mut dst_stage_mask: u32 = vk::PipelineStageFlags::BOTTOM_OF_PIPE.as_raw();
        let memory_barrier = if let Some(memory_barrier) = memory_barrier {
            let (src, dst, mb) = memory_barrier.to_vk();
            src_stage_mask |= src.as_raw();
            dst_stage_mask |= dst.as_raw();
            Some(mb)
        } else {
            None
        };
        let buffer_barriers = unsafe {
            let mut v: [vk::BufferMemoryBarrier; BL] = [MaybeUninit::uninit().assume_init(); BL];
            let mut i = 0;
            while i < buffer_barriers.len() {
                let buffer_barrier = &buffer_barriers[i];
                let (src, dst, mb) = buffer_barrier.to_vk();
                src_stage_mask |= src.as_raw();
                dst_stage_mask |= dst.as_raw();
                v[i] = mb;
                i += 1;
            }
            v
        };
        let image_barriers = unsafe {
            let mut v = [MaybeUninit::uninit().assume_init(); IL];
            let mut i = 0;
            while i < image_barriers.len() {
                let image_barrier = &image_barriers[i];
                let (src, dst, mb) = image_barrier.to_vk();
                src_stage_mask |= src.as_raw();
                dst_stage_mask |= dst.as_raw();
                v[i] = mb;
                i += 1;
            }
            v
        };
        Self {
            src_stage_mask: vk::PipelineStageFlags::from_raw(src_stage_mask),
            dst_stage_mask: vk::PipelineStageFlags::from_raw(dst_stage_mask),
            dependency_flag,
            memory_barrier,
            buffer_barriers,
            image_barriers,
        }
    }
}

pub struct PipelineBarrier {
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flag: vk::DependencyFlags,
    memory_barrier: Option<vk::MemoryBarrier>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier>,
    image_barriers: Vec<vk::ImageMemoryBarrier>,
}

impl PipelineBarrier {
    pub fn new(
        memory_barrier: Option<MemoryBarrier>,
        buffer_barriers: &[BufferBarrier],
        image_barriers: &[ImageBarrier],
        dependency_flag: vk::DependencyFlags,
    ) -> Self {
        let mut src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
        let mut dst_stage_mask = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        let memory_barrier = if let Some(memory_barrier) = memory_barrier {
            let (src, dst, mb) = memory_barrier.to_vk();
            src_stage_mask |= src;
            dst_stage_mask |= dst;
            Some(mb)
        } else {
            None
        };
        let buffer_barriers = {
            let (src, dst, v) = buffer_barriers.iter().map(|a| a.to_vk()).fold(
                (
                    vk::PipelineStageFlags::empty(),
                    vk::PipelineStageFlags::empty(),
                    Vec::with_capacity(buffer_barriers.len()),
                ),
                |a, b| {
                    let mut v = a.2;
                    v.push(b.2);
                    (a.0 | b.0, a.1 | b.1, v)
                },
            );
            src_stage_mask |= src;
            dst_stage_mask |= dst;
            v
        };
        let image_barriers = {
            let (src, dst, v) = image_barriers.iter().map(|a| a.to_vk()).fold(
                (
                    vk::PipelineStageFlags::empty(),
                    vk::PipelineStageFlags::empty(),
                    Vec::with_capacity(image_barriers.len()),
                ),
                |a, b| {
                    let mut v = a.2;
                    v.push(b.2);
                    (a.0 | b.0, a.1 | b.1, v)
                },
            );
            src_stage_mask |= src;
            dst_stage_mask |= dst;
            v
        };
        Self {
            src_stage_mask,
            dst_stage_mask,
            dependency_flag,
            memory_barrier,
            buffer_barriers,
            image_barriers,
        }
    }
}

impl<'a> CommandRecorder<'a> {
    /// Insert a memory dependency.
    /// Pipeline Barrier parameters are generated at compile time.
    pub fn simple_const_pipeline_barrier<const BL: usize, const IL: usize>(
        &mut self,
        barrier: &PipelineBarrierConst<BL, IL>,
    ) -> &mut Self {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                barrier.src_stage_mask,
                barrier.dst_stage_mask,
                barrier.dependency_flag,
                barrier
                    .memory_barrier
                    .as_ref()
                    .map_or(&[], |mb| std::slice::from_ref(mb)),
                barrier.buffer_barriers.as_ref(),
                barrier.image_barriers.as_ref(),
            )
        }
        self
    }

    /// Insert a memory dependency.
    pub fn simple_pipeline_barrier(&mut self, barrier: &PipelineBarrier) -> &mut Self {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                barrier.src_stage_mask,
                barrier.dst_stage_mask,
                barrier.dependency_flag,
                barrier
                    .memory_barrier
                    .as_ref()
                    .map_or(&[], |mb| std::slice::from_ref(mb)),
                barrier.buffer_barriers.as_ref(),
                barrier.image_barriers.as_ref(),
            )
        }
        self
    }
}
