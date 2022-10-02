use ash::vk;

pub(crate) fn shader_stage_to_pipeline_stage(
    shader_stage_flags: vk::ShaderStageFlags,
) -> vk::PipelineStageFlags2 {
    match shader_stage_flags {
        vk::ShaderStageFlags::ANY_HIT_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::MISS_KHR
        | vk::ShaderStageFlags::CALLABLE_KHR
        | vk::ShaderStageFlags::INTERSECTION_KHR => vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        vk::ShaderStageFlags::FRAGMENT => vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::ShaderStageFlags::VERTEX => vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::ShaderStageFlags::COMPUTE => vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::ShaderStageFlags::GEOMETRY => vk::PipelineStageFlags2::GEOMETRY_SHADER,
        vk::ShaderStageFlags::MESH_NV => vk::PipelineStageFlags2::MESH_SHADER_NV,
        vk::ShaderStageFlags::TESSELLATION_CONTROL => {
            vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER
        }
        vk::ShaderStageFlags::TESSELLATION_EVALUATION => {
            vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER
        }
        _ => vk::PipelineStageFlags2::empty(),
    }
}

pub(crate) fn descriptor_type_to_access_flags_read(
    descriptor_type: vk::DescriptorType,
) -> vk::AccessFlags2 {
    match descriptor_type {
        vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
        }
        vk::DescriptorType::STORAGE_IMAGE
        | vk::DescriptorType::STORAGE_BUFFER
        | vk::DescriptorType::STORAGE_TEXEL_BUFFER
        | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => vk::AccessFlags2::SHADER_STORAGE_READ,
        vk::DescriptorType::UNIFORM_BUFFER
        | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        | vk::DescriptorType::UNIFORM_TEXEL_BUFFER => vk::AccessFlags2::UNIFORM_READ,
        vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
            vk::AccessFlags2::SHADER_SAMPLED_READ
        }
        _ => vk::AccessFlags2::empty(),
    }
}
pub(crate) fn descriptor_type_to_access_flags_write(
    descriptor_type: vk::DescriptorType,
) -> vk::AccessFlags2 {
    match descriptor_type {
        vk::DescriptorType::STORAGE_IMAGE
        | vk::DescriptorType::STORAGE_BUFFER
        | vk::DescriptorType::STORAGE_TEXEL_BUFFER
        | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => vk::AccessFlags2::SHADER_STORAGE_WRITE,
        _ => vk::AccessFlags2::empty(),
    }
}

pub mod pipline_stage_order {
    use ash::vk;
    use vk::PipelineStageFlags2 as F;

    const FRAGMENT_BITS: vk::PipelineStageFlags2 = F::from_raw(
        F::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR.as_raw()
            | F::EARLY_FRAGMENT_TESTS.as_raw()
            | F::FRAGMENT_SHADER.as_raw()
            | F::LATE_FRAGMENT_TESTS.as_raw()
            | F::COLOR_ATTACHMENT_OUTPUT.as_raw(),
    );
    const GRAPHICS_BITS: vk::PipelineStageFlags2 = F::from_raw(
        F::DRAW_INDIRECT.as_raw()
            | F::INDEX_INPUT.as_raw()
            | F::VERTEX_ATTRIBUTE_INPUT.as_raw()
            | F::VERTEX_SHADER.as_raw()
            | F::TESSELLATION_CONTROL_SHADER.as_raw()
            | F::TESSELLATION_EVALUATION_SHADER.as_raw()
            | F::GEOMETRY_SHADER.as_raw()
            | F::TRANSFORM_FEEDBACK_EXT.as_raw()
            | FRAGMENT_BITS.as_raw(),
    );

    const GRAPHICS_MESH_BITS: vk::PipelineStageFlags2 = F::from_raw(
        F::DRAW_INDIRECT.as_raw()
            | F::TASK_SHADER_NV.as_raw()
            | F::MESH_SHADER_NV.as_raw()
            | FRAGMENT_BITS.as_raw(),
    );

    const COMPUTE_BITS: vk::PipelineStageFlags2 =
        F::from_raw(F::DRAW_INDIRECT.as_raw() | F::COMPUTE_SHADER.as_raw());

    const RTX_BITS: vk::PipelineStageFlags2 =
        F::from_raw(F::DRAW_INDIRECT.as_raw() | F::RAY_TRACING_SHADER_KHR.as_raw());
    /// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-order
    pub fn logically_later_stages(stage: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
        match stage {
            F::DRAW_INDIRECT => GRAPHICS_BITS | GRAPHICS_MESH_BITS | COMPUTE_BITS | RTX_BITS,
            F::INDEX_INPUT => GRAPHICS_BITS & !F::DRAW_INDIRECT,
            F::VERTEX_ATTRIBUTE_INPUT => GRAPHICS_BITS & !(F::DRAW_INDIRECT | F::INDEX_INPUT),
            F::VERTEX_SHADER => {
                GRAPHICS_BITS & !(F::DRAW_INDIRECT | F::INDEX_INPUT | F::VERTEX_ATTRIBUTE_INPUT)
            }
            F::TESSELLATION_CONTROL_SHADER => {
                F::TESSELLATION_CONTROL_SHADER
                    | F::TESSELLATION_EVALUATION_SHADER
                    | F::GEOMETRY_SHADER
                    | F::TRANSFORM_FEEDBACK_EXT
                    | FRAGMENT_BITS
            }

            F::TESSELLATION_EVALUATION_SHADER => {
                F::TESSELLATION_EVALUATION_SHADER
                    | F::GEOMETRY_SHADER
                    | F::TRANSFORM_FEEDBACK_EXT
                    | FRAGMENT_BITS
            }
            F::GEOMETRY_SHADER => F::GEOMETRY_SHADER | F::TRANSFORM_FEEDBACK_EXT | FRAGMENT_BITS,
            F::TRANSFORM_FEEDBACK_EXT => F::TRANSFORM_FEEDBACK_EXT | FRAGMENT_BITS,
            F::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR => FRAGMENT_BITS,
            F::EARLY_FRAGMENT_TESTS => FRAGMENT_BITS & !(F::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR),
            F::FRAGMENT_SHADER => {
                F::FRAGMENT_SHADER | F::LATE_FRAGMENT_TESTS | F::COLOR_ATTACHMENT_OUTPUT
            }
            F::LATE_FRAGMENT_TESTS => F::LATE_FRAGMENT_TESTS | F::COLOR_ATTACHMENT_OUTPUT,
            F::COLOR_ATTACHMENT_OUTPUT => F::COLOR_ATTACHMENT_OUTPUT,

            F::TASK_SHADER_NV => F::TASK_SHADER_NV | F::MESH_SHADER_NV | FRAGMENT_BITS,
            F::MESH_SHADER_NV => F::MESH_SHADER_NV | FRAGMENT_BITS,

            F::FRAGMENT_DENSITY_PROCESS_EXT => {
                F::FRAGMENT_DENSITY_PROCESS_EXT | F::EARLY_FRAGMENT_TESTS
            }

            _ => stage,
        }
    }

    pub fn access_is_write(access: vk::AccessFlags2) -> bool {
        use vk::AccessFlags2 as F;
        access.intersects(
            F::SHADER_WRITE
                | F::COLOR_ATTACHMENT_WRITE
                | F::DEPTH_STENCIL_ATTACHMENT_WRITE
                | F::TRANSFER_WRITE
                | F::HOST_WRITE
                | F::MEMORY_WRITE
                | F::SHADER_STORAGE_WRITE,
        )
    }
}
