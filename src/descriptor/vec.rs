use std::{ffi::c_void, mem::MaybeUninit, sync::Arc};

use crate::{command::pool, Device, HasDevice};

use super::{DescriptorPool, DescriptorSet, DescriptorSetLayout};
use ash::{prelude::VkResult, vk};
/// Dynamically resized descriptor pool, useful for bindless
pub struct DescriptorVec {
    /// self.capacity[binding_type] = number of descriptors currently allocated for that type
    capacity: [u32; 4],
    /// Total number of descriptors currently allocated.
    sizes: [u32; 4],

    /// descriptors at the following locations are considered free
    freelists: [Vec<u32>; 4],
    /// All descriptors after these indices are considered free
    tail: [u32; 4],

    pool: Arc<DescriptorPool>,
    layout: DescriptorSetLayout,
    desc: Arc<DescriptorSet>,
    shader_stage_flags: vk::ShaderStageFlags,
}

#[derive(Clone, Copy)]
enum DescriptorVecBindingType {
    SampledImage = 0,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
}

pub enum DescriptorVecBinding {
    SampledImage(vk::DescriptorImageInfo),
    StorageImage(vk::DescriptorImageInfo),
    UniformBuffer(vk::DescriptorBufferInfo),
    StorageBuffer(vk::DescriptorBufferInfo),
}
impl DescriptorVecBinding {
    fn ty(&self) -> DescriptorVecBindingType {
        match self {
            Self::SampledImage(_) => DescriptorVecBindingType::SampledImage,
            Self::StorageImage(_) => DescriptorVecBindingType::StorageImage,
            Self::UniformBuffer(_) => DescriptorVecBindingType::UniformBuffer,
            Self::StorageBuffer(_) => DescriptorVecBindingType::StorageBuffer,
        }
    }
}

impl TryFrom<u32> for DescriptorVecBindingType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SampledImage),
            1 => Ok(Self::StorageImage),
            2 => Ok(Self::UniformBuffer),
            3 => Ok(Self::StorageBuffer),
            _ => Err(()),
        }
    }
}
impl From<DescriptorVecBindingType> for vk::DescriptorType {
    fn from(ty: DescriptorVecBindingType) -> Self {
        match ty {
            DescriptorVecBindingType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            DescriptorVecBindingType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            DescriptorVecBindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            DescriptorVecBindingType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

impl DescriptorVec {
    pub fn create_layout(
        device: Arc<Device>,
        binding_counts: &[u32; 4],
        shader_stage_flags: vk::ShaderStageFlags,
    ) -> VkResult<DescriptorSetLayout> {
        let mut bindings: [MaybeUninit<vk::DescriptorSetLayoutBinding>; 4] =
            MaybeUninit::uninit_array();
        for (i, item) in bindings.iter_mut().enumerate() {
            item.write(vk::DescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: DescriptorVecBindingType::try_from(i as u32).unwrap().into(),
                descriptor_count: binding_counts[i],
                stage_flags: shader_stage_flags,
                p_immutable_samplers: std::ptr::null(),
            });
        }
        let bindings: [vk::DescriptorSetLayoutBinding; 4] =
            unsafe { std::mem::transmute(bindings) };

        let flags = [vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND; 4];
        let flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
            binding_count: binding_counts.len() as u32,
            p_binding_flags: flags.as_ptr(),
            ..Default::default()
        };
        unsafe {
            DescriptorSetLayout::new(
                device,
                &vk::DescriptorSetLayoutCreateInfo {
                    p_next: &flags as *const _ as *const c_void,
                    flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                    binding_count: binding_counts.len() as u32,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                },
            )
        }
    }
    pub fn create_pool(device: Arc<Device>, binding_counts: &[u32; 4]) -> VkResult<DescriptorPool> {
        let mut pool_sizes: [MaybeUninit<vk::DescriptorPoolSize>; 4] = MaybeUninit::uninit_array();
        for (i, item) in pool_sizes.iter_mut().enumerate() {
            item.write(vk::DescriptorPoolSize {
                ty: DescriptorVecBindingType::try_from(i as u32).unwrap().into(),
                descriptor_count: binding_counts[i],
            });
        }
        let pool_sizes: [vk::DescriptorPoolSize; 4] = unsafe { std::mem::transmute(pool_sizes) };
        let info = vk::DescriptorPoolCreateInfo {
            flags: vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            max_sets: binding_counts.len() as u32,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        DescriptorPool::new(device, &info)
    }
    pub fn new(device: Arc<Device>, shader_stage_flags: vk::ShaderStageFlags) -> VkResult<Self> {
        let default_binding_counts: [u32; 4] = [4, 4, 4, 4];
        let layout =
            Self::create_layout(device.clone(), &default_binding_counts, shader_stage_flags)?;
        let pool = Self::create_pool(device, &default_binding_counts)?;
        let pool = Arc::new(pool);
        let mut desc = pool.allocate(std::iter::once(&layout))?;
        assert_eq!(desc.len(), 1);
        let desc = desc.drain(..).next().unwrap();
        Ok(Self {
            capacity: default_binding_counts,
            sizes: [0; 4],
            freelists: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            tail: [0; 4],
            pool,
            desc: Arc::new(desc),
            shader_stage_flags,
            layout,
        })
    }

    pub fn realloc(&mut self, new_capacity: [u32; 4]) -> VkResult<()> {
        let device = self.pool.device().clone();
        let layout = Self::create_layout(device.clone(), &new_capacity, self.shader_stage_flags)?;
        let new_pool = Self::create_pool(device, &new_capacity)?;
        let new_pool = Arc::new(new_pool);
        let mut new_desc = new_pool.allocate(std::iter::once(&layout))?;
        assert_eq!(new_desc.len(), 1);
        let new_desc = new_desc.drain(..).next().unwrap();

        {
            let mut copy_desc_sets: [MaybeUninit<vk::CopyDescriptorSet>; 4] =
                MaybeUninit::uninit_array();
            for (i, item) in copy_desc_sets.iter_mut().enumerate() {
                item.write(vk::CopyDescriptorSet {
                    src_set: self.desc.raw,
                    src_binding: 0,
                    src_array_element: 0,
                    dst_set: new_desc.raw,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: self.capacity[i],
                    ..Default::default()
                });
            }

            unsafe {
                let copy_desc_sets: [vk::CopyDescriptorSet; 4] =
                    std::mem::transmute(copy_desc_sets);
                new_pool
                    .device()
                    .update_descriptor_sets(&[], &copy_desc_sets);
            }
        }

        self.capacity = new_capacity;
        self.desc = Arc::new(new_desc);
        self.pool = new_pool;
        self.layout = layout;
        Ok(())
    }

    pub fn extend(
        &mut self,
        items: impl IntoIterator<Item = DescriptorVecBinding>,
    ) -> VkResult<Vec<u32>> {
        let mut num_desc_to_allocate: [u32; 4] = [0; 4];

        let mut descriptor_image_info: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut descriptor_buffer_info: Vec<vk::DescriptorBufferInfo> = Vec::new();

        let mut writes: Vec<_> = items
            .into_iter()
            .map(|binding| {
                let descriptor_type = binding.ty();
                num_desc_to_allocate[descriptor_type as usize] += 1;
                match binding {
                    DescriptorVecBinding::SampledImage(i)
                    | DescriptorVecBinding::StorageImage(i) => descriptor_image_info.push(i),
                    DescriptorVecBinding::UniformBuffer(i)
                    | DescriptorVecBinding::StorageBuffer(i) => descriptor_buffer_info.push(i),
                }
                vk::WriteDescriptorSet {
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: descriptor_type as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: descriptor_type.into(),
                    ..Default::default()
                }
            })
            .collect();

        let size_after_alloc = self.sizes.zip(num_desc_to_allocate).map(|(a, b)| a + b);
        let needs_realloc = size_after_alloc
            .iter()
            .enumerate()
            .any(|(i, &size)| size > self.capacity[i]);
        if needs_realloc {
            let needed_size =
                size_after_alloc
                    .zip(self.capacity)
                    .map(|(size_after_alloc, capacity)| {
                        if size_after_alloc > capacity {
                            size_after_alloc * 2
                        } else {
                            capacity
                        }
                    });
            self.realloc(needed_size)?;
        }

        let mut image_info_indice: usize = 0;
        let mut buffer_info_indice: usize = 0;
        let mut array_elemnets: Vec<u32> = Vec::with_capacity(writes.len());
        for write in writes.iter_mut() {
            write.dst_set = self.desc.raw;
            let ty = write.dst_binding;
            write.dst_array_element = if let Some(element) = self.freelists[ty as usize].pop() {
                element
            } else {
                let element = self.tail[ty as usize];
                self.tail[ty as usize] += 1;
                element
            };
            array_elemnets.push(write.dst_array_element);
            if write.descriptor_type == vk::DescriptorType::STORAGE_IMAGE
                || write.descriptor_type == vk::DescriptorType::SAMPLED_IMAGE
            {
                write.p_image_info = &descriptor_image_info[image_info_indice];
                image_info_indice += 1;
            } else if write.descriptor_type == vk::DescriptorType::STORAGE_BUFFER
                || write.descriptor_type == vk::DescriptorType::UNIFORM_BUFFER
            {
                write.p_buffer_info = &descriptor_buffer_info[buffer_info_indice];
                buffer_info_indice += 1;
            } else {
                panic!()
            }
        }
        unsafe {
            self.pool.device().update_descriptor_sets(&writes, &[]);
        }
        Ok(array_elemnets)
    }

    pub fn raw(&self) -> vk::DescriptorSet {
        self.desc.raw
    }
    pub fn raw_layout(&self) -> vk::DescriptorSetLayout {
        self.layout.raw
    }
}
