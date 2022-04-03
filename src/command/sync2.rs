use std::{mem::MaybeUninit, ptr::null};

use ash::vk;

use super::recorder::CommandRecorder;
pub use super::sync::{AccessType, BufferBarrier, ImageBarrier, ImageLayout, MemoryBarrier};
struct VkAccessInfo2 {
    stage_mask: vk::PipelineStageFlags2,
    access_mask: vk::AccessFlags2,
    image_layout: vk::ImageLayout,
}

impl AccessType {
    const fn to_vk2(self) -> VkAccessInfo2 {
        match self {
            AccessType::CommandBufferReadNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COMMAND_PREPROCESS_NV,
                access_mask: vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::IndirectBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT,
                access_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::IndexBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::VERTEX_INPUT,
                access_mask: vk::AccessFlags2::INDEX_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::VERTEX_INPUT,
                access_mask: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderReadUniformBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::VertexShaderReadOther => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationControlShaderReadUniformBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer => {
                VkAccessInfo2 {
                    stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                    access_mask: vk::AccessFlags2::SHADER_READ,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }
            }
            AccessType::TessellationControlShaderReadOther => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationEvaluationShaderReadUniformBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer => {
                VkAccessInfo2 {
                    stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                    access_mask: vk::AccessFlags2::SHADER_READ,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }
            }
            AccessType::TessellationEvaluationShaderReadOther => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderReadUniformBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::GeometryShaderReadOther => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TaskShaderReadUniformBufferNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TaskShaderReadSampledImageOrUniformTexelBufferNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::TaskShaderReadOtherNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::MeshShaderReadUniformBufferNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::MeshShaderReadSampledImageOrUniformTexelBufferNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::MeshShaderReadOtherNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransformFeedbackCounterReadEXT => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentDensityMapReadEXT => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT,
                access_mask: vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT,
                image_layout: vk::ImageLayout::FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
            },
            AccessType::ShadingRateReadNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::SHADING_RATE_IMAGE_NV,
                access_mask: vk::AccessFlags2::SHADING_RATE_IMAGE_READ_NV,
                image_layout: vk::ImageLayout::SHADING_RATE_OPTIMAL_NV,
            },
            AccessType::FragmentShaderReadUniformBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadColorInputAttachment => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadDepthStencilInputAttachment => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadOther => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ColorAttachmentRead => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::ColorAttachmentAdvancedBlendingEXT => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentRead => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
            AccessType::ComputeShaderReadUniformBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::ComputeShaderReadOther => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderReadUniformBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::UNIFORM_READ,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AnyShaderReadUniformBufferOrVertexBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::UNIFORM_READ.as_raw()
                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw(),
                ),
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::AnyShaderReadOther => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::SHADER_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransferRead => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ALL_TRANSFER,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::CopyRead => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COPY,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::BlitRead => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::BLIT,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::ResolveRead => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::RESOLVE,
                access_mask: vk::AccessFlags2::TRANSFER_READ,
                image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::HostRead => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::HOST,
                access_mask: vk::AccessFlags2::HOST_READ,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::Present => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::empty(),
                access_mask: vk::AccessFlags2::empty(),
                image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
            AccessType::ConditionalRenderingReadEXT => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::CONDITIONAL_RENDERING_EXT,
                access_mask: vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::RayTracingShaderAccelerationStructureReadKHR => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AccelerationStructureBuildReadKHR => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::CommandBufferWriteNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COMMAND_PREPROCESS_NV,
                access_mask: vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationControlShaderWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessellationEvaluationShaderWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::GEOMETRY_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TaskShaderWriteNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TASK_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::MeshShaderWriteNV => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::MESH_SHADER_NV,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransformFeedbackWriteEXT => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TransformFeedbackCounterWriteEXT => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
                access_mask: vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentShaderWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ColorAttachmentWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                ),
                access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthAttachmentWriteStencilReadOnly => VkAccessInfo2 {
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
            AccessType::StencilAttachmentWriteDepthReadOnly => VkAccessInfo2 {
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
            AccessType::ComputeShaderWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                access_mask: vk::AccessFlags2::SHADER_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransferWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ALL_TRANSFER,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::CopyWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COPY,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::BlitWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::BLIT,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::ResolveWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::RESOLVE,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::ClearWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::CLEAR,
                access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            AccessType::HostPreinitialized => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::HOST,
                access_mask: vk::AccessFlags2::HOST_WRITE,
                image_layout: vk::ImageLayout::PREINITIALIZED,
            },
            AccessType::HostWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::HOST,
                access_mask: vk::AccessFlags2::HOST_WRITE,
                image_layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AccelerationStructureBuildWriteKHR => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                access_mask: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                image_layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::ColorAttachmentReadWrite => VkAccessInfo2 {
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw(),
                ),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::General => VkAccessInfo2 {
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
    const fn to_vk2(&self) -> vk::MemoryBarrier2 {
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
            let info = prev_access.to_vk2();
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
            let info = next_access.to_vk2();
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
    const fn to_vk2(&self) -> vk::BufferMemoryBarrier2 {
        assert!(self.src_queue_family_index != self.dst_queue_family_index, "BufferBarrier should only be used when a queue family ownership transfer is required. Use MemoryBarrier instead.");
        let barrier = self.memory_barrier.to_vk2();
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
    const fn to_vk2(&self) -> vk::ImageMemoryBarrier2 {
        let barrier = self.memory_barrier.to_vk2();
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
                let info = prev_access.to_vk2();
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
                let info = next_access.to_vk2();

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

pub struct PipelineBarrierConst<const BL: usize, const IL: usize> {
    dependency_flag: vk::DependencyFlags,
    memory_barrier: Option<vk::MemoryBarrier2>,
    buffer_barriers: [vk::BufferMemoryBarrier2; BL],
    image_barriers: [vk::ImageMemoryBarrier2; IL],
}

impl<const BL: usize, const IL: usize> PipelineBarrierConst<BL, IL> {
    pub const fn new(
        memory_barrier: Option<MemoryBarrier>,
        buffer_barriers: &[BufferBarrier; BL],
        image_barriers: &[ImageBarrier; IL],
        dependency_flag: vk::DependencyFlags,
    ) -> Self {
        let memory_barrier = memory_barrier.as_ref().map(MemoryBarrier::to_vk2);
        let buffer_barriers = unsafe {
            let mut v: [vk::BufferMemoryBarrier2; BL] = [MaybeUninit::uninit().assume_init(); BL];
            let mut i = 0;
            while i < buffer_barriers.len() {
                let buffer_barrier = &buffer_barriers[i];
                v[i] = buffer_barrier.to_vk2();
                i += 1;
            }
            v
        };
        let image_barriers = unsafe {
            let mut v = [MaybeUninit::uninit().assume_init(); IL];
            let mut i = 0;
            while i < image_barriers.len() {
                let image_barrier = &image_barriers[i];
                v[i] = image_barrier.to_vk2();
                i += 1;
            }
            v
        };
        Self {
            dependency_flag,
            memory_barrier,
            buffer_barriers,
            image_barriers,
        }
    }

    const fn to_dependency_info(&self) -> vk::DependencyInfo {
        #[inline]
        const fn ref_to_ptr<T>(a: &T) -> *const T {
            a
        }
        vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            p_next: null(),
            dependency_flags: self.dependency_flag,
            memory_barrier_count: if self.memory_barrier.is_some() { 1 } else { 0 },
            p_memory_barriers: self
                .memory_barrier
                .as_ref()
                .map(ref_to_ptr)
                .unwrap_or(null()),
            buffer_memory_barrier_count: self.buffer_barriers.len() as u32,
            p_buffer_memory_barriers: self.buffer_barriers.as_ptr(),
            image_memory_barrier_count: self.image_barriers.len() as u32,
            p_image_memory_barriers: self.image_barriers.as_ptr(),
        }
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
        let memory_barrier = memory_barrier.as_ref().map(MemoryBarrier::to_vk2);
        let buffer_barriers = buffer_barriers.iter().map(BufferBarrier::to_vk2).collect();
        let image_barriers = image_barriers.iter().map(ImageBarrier::to_vk2).collect();
        Self {
            dependency_flag,
            memory_barrier,
            buffer_barriers,
            image_barriers,
        }
    }
    fn to_dependency_info(&self) -> vk::DependencyInfo {
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
    /// Pipeline Barrier parameters are generated at compile time.
    pub fn simple_const_pipeline_barrier2<const BL: usize, const IL: usize>(
        &mut self,
        barrier: &PipelineBarrierConst<BL, IL>,
    ) -> &mut Self {
        let dep_info = barrier.to_dependency_info();
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer, &dep_info)
        }
        self
    }

    /// Insert a memory dependency.
    pub fn simple_pipeline_barrier2(&mut self, barrier: &PipelineBarrier) -> &mut Self {
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
        .to_vk2();
        assert_eq!(barrier.src_stage_mask, vk::PipelineStageFlags2::TOP_OF_PIPE);
        assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(barrier.src_access_mask, vk::AccessFlags2::NONE);
        assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::TRANSFER_WRITE);
    }
}
