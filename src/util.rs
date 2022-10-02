use ash::vk;

pub(crate) fn shader_stage_to_pipeline_stage(shader_stage_flags: vk::ShaderStageFlags) -> vk::PipelineStageFlags2 {
    match shader_stage_flags {
        vk::ShaderStageFlags::ANY_HIT_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR |
        vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::MISS_KHR |
        vk::ShaderStageFlags::CALLABLE_KHR | vk::ShaderStageFlags::INTERSECTION_KHR  => vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        vk::ShaderStageFlags::FRAGMENT => vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::ShaderStageFlags::VERTEX => vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::ShaderStageFlags::COMPUTE => vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::ShaderStageFlags::GEOMETRY => vk::PipelineStageFlags2::GEOMETRY_SHADER,
        vk::ShaderStageFlags::MESH_NV => vk::PipelineStageFlags2::MESH_SHADER_NV,
        vk::ShaderStageFlags::TESSELLATION_CONTROL => vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
        vk::ShaderStageFlags::TESSELLATION_EVALUATION => vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
        _ => vk::PipelineStageFlags2::empty()
    }
}

pub(crate) fn descriptor_type_to_access_flags_read(descriptor_type: vk::DescriptorType) -> vk::AccessFlags2 {
    match descriptor_type {
        vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::STORAGE_TEXEL_BUFFER |
        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => vk::AccessFlags2::SHADER_STORAGE_READ,
        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC | vk::DescriptorType::UNIFORM_TEXEL_BUFFER => vk::AccessFlags2::UNIFORM_READ,
        vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::COMBINED_IMAGE_SAMPLER => vk::AccessFlags2::SHADER_SAMPLED_READ,
        _ => vk::AccessFlags2::empty()
    }
}
pub(crate) fn descriptor_type_to_access_flags_write(descriptor_type: vk::DescriptorType) -> vk::AccessFlags2 {
    match descriptor_type {
        vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::STORAGE_TEXEL_BUFFER |
        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => vk::AccessFlags2::SHADER_STORAGE_WRITE,
        _ => vk::AccessFlags2::empty()
    }
}
