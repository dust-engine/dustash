use std::{mem::MaybeUninit, ptr::null};

use ash::vk;

use super::recorder::CommandRecorder;


#[derive(Copy, Clone, Debug)]
pub enum AccessType {
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
    /// Read as the source of a copy operation
    CopyRead,
    /// Read as the source of a blit operation
    BlitRead,
    /// Read as the source of a resolve operation
    ResolveRead,

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
    /// Written as any resource in a ray tracing shader
    RayTracingShaderWrite,
    /// Written as any resource in any shader
    AnyShaderWrite,

    /// Written as the destination of a transfer operation
    TransferWrite,
    /// Written as the target of a copy operation
    CopyWrite,
    /// Written as the target of a blit operation
    BlitWrite,
    /// Written as the target of a resolve operation
    ResolveWrite,
    /// Written as the destination of a clear operation, with the exception of vkCmdClearAttachments.
    ClearWrite,

    /// Data pre-filled by host before device access starts
    HostPreinitialized,
    /// Written on the host
    HostWrite,

    /// Written as an acceleration structure during a build.
    /// Requires VK_KHR_acceleration_structure to be enabled.
    AccelerationStructureBuildWriteKHR,

    /// Read or written as a color attachment during rendering
    ColorAttachmentReadWrite,
    // General access
    /// Covers any access - useful for debug, generally avoid for performance reasons
    General,
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

/// Global barriers define a set of accesses on multiple resources at once.
/// If a buffer or image doesn't require a queue ownership transfer, or an image
/// doesn't require a layout transition (e.g. you're using one of the GENERAL
/// layouts) then a global barrier should be preferred.
/// Simply define the previous and next access types of resources affected.
pub struct MemoryBarrier<'a> {
    pub prev_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
}
impl<'a> Default for MemoryBarrier<'a> {
    fn default() -> Self {
        Self {
            prev_accesses: &[],
            next_accesses: &[],
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


struct VkAccessInfo {
    stage_mask: vk::PipelineStageFlags2,
    access_mask: vk::AccessFlags2,
    image_layout: vk::ImageLayout,
}

impl AccessType {
        pub const fn is_read_only(&self) -> bool {
        let this = *self as u32;
        this <= (Self::Present as u32)
    }
    pub const fn is_write(&self) -> bool {
        (*self as u32) > (Self::Present as u32)
    }

    const fn to_vk(self) -> VkAccessInfo {
        match self {
            AccessType::CommandBufferReadNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COMMAND_PREPROCESS_NV,
                access_mask: vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::IndirectBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT,
                access_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::IndexBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::VERTEX_INPUT,
                access_mask: vk::AccessFlags2::INDEX_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::VERTEX_INPUT,
                access_mask: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::VertexShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationControlShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer => {
                VkAccessInfo {
                    stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                    access_mask: vk::AccessFlags2::SHADER_READ,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }
            }
            AccessType::TessellationControlShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationEvaluationShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer => {
                VkAccessInfo {
                    stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                    access_mask: vk::AccessFlags2::SHADER_READ,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }
            }
            AccessType::TessellationEvaluationShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::GeometryShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TaskShaderReadUniformBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TaskShaderReadSampledImageOrUniformTexelBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::TaskShaderReadOtherNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::MeshShaderReadUniformBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::MeshShaderReadSampledImageOrUniformTexelBufferNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::MeshShaderReadOtherNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransformFeedbackCounterReadEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentDensityMapReadEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT,
                access_mask: vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT,
                image_layout: vk::ImageLayout::FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
            },
            AccessType::ShadingRateReadNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::SHADING_RATE_IMAGE_NV,
                access_mask: vk::AccessFlags2::SHADING_RATE_IMAGE_READ_NV,
                image_layout: vk::ImageLayout::SHADING_RATE_OPTIMAL_NV,
            },
            AccessType::FragmentShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadColorInputAttachment => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadDepthStencilInputAttachment => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ColorAttachmentRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::ColorAttachmentAdvancedBlendingEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            AccessType::ComputeShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::ComputeShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderReadUniformBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AnyShaderReadUniformBufferOrVertexBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::UNIFORM_READ.as_raw()
                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw(),
                ),
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::AnyShaderReadOther => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransferRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_TRANSFER,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::CopyRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COPY,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::BlitRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::BLIT,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::ResolveRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::RESOLVE,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::HostRead => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::HOST,
                access_mask: vk::AccessFlags2::HOST_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::Present => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::empty(),
                access_mask: vk::AccessFlags2::empty(),
                image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
            AccessType::ConditionalRenderingReadEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::CONDITIONAL_RENDERING_EXT,
                access_mask: vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::RayTracingShaderAccelerationStructureReadKHR => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AccelerationStructureBuildReadKHR => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::CommandBufferWriteNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COMMAND_PREPROCESS_NV,
                access_mask: vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationControlShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationEvaluationShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TaskShaderWriteNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::MeshShaderWriteNV => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransformFeedbackWriteEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TransformFeedbackCounterWriteEXT => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ColorAttachmentWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthAttachmentWriteStencilReadOnly => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw(),
                ),
                image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
            },
            AccessType::StencilAttachmentWriteDepthReadOnly => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw(),
                ),
                image_layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
            },
            AccessType::ComputeShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::RayTracingShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransferWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_TRANSFER,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::CopyWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COPY,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::BlitWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::BLIT,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::ResolveWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::RESOLVE,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::ClearWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::CLEAR,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::HostPreinitialized => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::HOST,
                access_mask: vk::AccessFlags2::HOST_WRITE,
                image_layout: vk::ImageLayout::PREINITIALIZED,
            },
            AccessType::HostWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::HOST,
                access_mask: vk::AccessFlags2::HOST_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AccelerationStructureBuildWriteKHR => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::ColorAttachmentReadWrite => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw(),
                ),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::General => VkAccessInfo {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::MEMORY_READ.as_raw()
                        | vk::AccessFlags2::MEMORY_WRITE.as_raw(),
                ),
                image_layout: vk::ImageLayout::GENERAL,
            },
        }
    }
}

/// Returns: (srcStages, dstStages)
impl<'a> MemoryBarrier<'a> {
    const fn to_vk(&self) -> vk::MemoryBarrier2 {
        let mut barrier = vk::MemoryBarrier2 {
            s_type: vk::StructureType::MEMORY_BARRIER_2,
            p_next: null(),
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::empty(),
            src_stage_mask: vk::PipelineStageFlags2::empty(),
            dst_stage_mask: vk::PipelineStageFlags2::empty(),
        };
        let mut src_stages = vk::PipelineStageFlags2::empty().as_raw();
        let mut dst_stages = vk::PipelineStageFlags2::empty().as_raw();
        let mut i: usize = 0;
        while i < self.prev_accesses.len() {
            let prev_access = &self.prev_accesses[i];
            let info = prev_access.to_vk();
            // Asserts that the access is a read, else it's a write and it should appear on its own.
            assert!(
                prev_access.is_read_only() || self.prev_accesses.len() == 1,
                "Multiple Writes"
            );
            src_stages |= info.stage_mask.as_raw();

            // Add appropriate availability operations - for writes only.
            if prev_access.is_write() {
                barrier.src_access_mask = vk::AccessFlags2::from_raw(
                    barrier.src_access_mask.as_raw() | info.access_mask.as_raw(),
                );
            }
            i += 1;
        }
        i = 0;
        while i < self.next_accesses.len() {
            let next_access = &self.next_accesses[i];
            let info = next_access.to_vk();
            assert!(
                next_access.is_read_only() || self.next_accesses.len() == 1,
                "Multiple Writes"
            );
            dst_stages |= info.stage_mask.as_raw();

            // Add visibility operations as necessary.
            // If the src access mask is zero, this is a WAR hazard (or for some reason a "RAR"),
            // so the dst access mask can be safely zeroed as these don't need visibility.
            if !barrier.src_access_mask.is_empty() {
                barrier.dst_access_mask = vk::AccessFlags2::from_raw(
                    barrier.dst_access_mask.as_raw() | info.access_mask.as_raw(),
                );
            }
            i += 1;
        }
        // Ensure that the stage masks are valid if no stages were determined
        barrier.src_stage_mask = vk::PipelineStageFlags2::from_raw(src_stages);
        barrier.dst_stage_mask = vk::PipelineStageFlags2::from_raw(dst_stages);
        if barrier.src_stage_mask.is_empty() {
            barrier.src_stage_mask = vk::PipelineStageFlags2::TOP_OF_PIPE;
        }
        if barrier.dst_stage_mask.is_empty() {
            barrier.dst_stage_mask = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
        }
        barrier
    }
}

impl<'a> BufferBarrier<'a> {
    const fn to_vk(&self) -> vk::BufferMemoryBarrier2 {
        assert!(self.src_queue_family_index != self.dst_queue_family_index, "BufferBarrier should only be used when a queue family ownership transfer is required. Use MemoryBarrier instead.");
        let barrier = self.memory_barrier.to_vk();
        vk::BufferMemoryBarrier2 {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
            p_next: null(),
            src_access_mask: barrier.src_access_mask,
            dst_access_mask: barrier.dst_access_mask,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
            src_stage_mask: barrier.src_stage_mask,
            dst_stage_mask: barrier.dst_stage_mask,
        }
    }
}

impl<'a> ImageBarrier<'a> {
    const fn to_vk(&self) -> vk::ImageMemoryBarrier2 {
        let barrier = self.memory_barrier.to_vk();
        let mut barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            p_next: null(),
            src_access_mask: barrier.src_access_mask,
            dst_access_mask: barrier.dst_access_mask,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            image: self.image,
            subresource_range: self.subresource_range,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::UNDEFINED,
            src_stage_mask: barrier.src_stage_mask,
            dst_stage_mask: barrier.dst_stage_mask,
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
                i += 1;
            }
        }

        {
            let mut i = 0;
            while i < self.memory_barrier.next_accesses.len() {
                let next_access = &self.memory_barrier.next_accesses[i];
                let info = next_access.to_vk();

                // neo: Additionally, if old layout is vk::ImageLayout::UNDEFINED,
                // dst_access_mask should still be added even if src_access_mask is empty.
                if barrier.old_layout.as_raw() == vk::ImageLayout::UNDEFINED.as_raw() {
                    barrier.dst_access_mask = vk::AccessFlags2::from_raw(
                        barrier.dst_access_mask.as_raw() | info.access_mask.as_raw(),
                    );
                }

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
                i += 1;
            }
        }

        assert!(barrier.new_layout.as_raw() != barrier.old_layout.as_raw() || barrier.src_queue_family_index != barrier.dst_queue_family_index, "Image barriers should only be used when a queue family ownership transfer or an image layout transition is required. Use MemoryBarrier instead.");

        barrier
    }
}

pub struct PipelineBarrier {
    dependency_flag: vk::DependencyFlags,
    memory_barrier: Option<vk::MemoryBarrier2>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
    image_barriers: Vec<vk::ImageMemoryBarrier2>,
}

impl PipelineBarrier {
    pub fn new(
        memory_barrier: Option<MemoryBarrier>,
        buffer_barriers: &[BufferBarrier],
        image_barriers: &[ImageBarrier],
        dependency_flag: vk::DependencyFlags,
    ) -> Self {
        let memory_barrier = memory_barrier.as_ref().map(MemoryBarrier::to_vk);
        let buffer_barriers = buffer_barriers.iter().map(BufferBarrier::to_vk).collect();
        let image_barriers = image_barriers.iter().map(ImageBarrier::to_vk).collect();
        Self {
            dependency_flag,
            memory_barrier,
            buffer_barriers,
            image_barriers,
        }
    }
    pub fn to_dependency_info(&self) -> vk::DependencyInfo {
        vk::DependencyInfo {
            dependency_flags: self.dependency_flag,
            memory_barrier_count: if self.memory_barrier.is_some() { 1 } else { 0 },
            p_memory_barriers: self
                .memory_barrier
                .as_ref()
                .map(|a| a as *const _)
                .unwrap_or(null()),
            buffer_memory_barrier_count: self.buffer_barriers.len() as u32,
            p_buffer_memory_barriers: self.buffer_barriers.as_ptr(),
            image_memory_barrier_count: self.image_barriers.len() as u32,
            p_image_memory_barriers: self.image_barriers.as_ptr(),
            ..Default::default()
        }
    }
}

impl<'a> CommandRecorder<'a> {
    /// Insert a memory dependency.
    pub fn simple_pipeline_barrier(&mut self, barrier: &PipelineBarrier) -> &mut Self {
        let dep_info = barrier.to_dependency_info();
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer, &dep_info)
        }
        self
    }
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn single_clear_img() {
        let barrier = ImageBarrier {
            memory_barrier: MemoryBarrier {
                prev_accesses: &[],
                next_accesses: &[AccessType::ClearWrite],
            },
            prev_layout: ImageLayout::Optimal,
            next_layout: ImageLayout::Optimal,
            discard_contents: true,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: vk::Image::null(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                base_mip_level: 0,
                level_count: 1,
            },
        }
        .to_vk();
        assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::TOP_OF_PIPE);
        assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(barrier.src_access_mask, vk::AccessFlags2::NONE);
        assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::TRANSFER_WRITE);
    }
}
