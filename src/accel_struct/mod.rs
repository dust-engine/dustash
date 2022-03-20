use std::{
    ops::{Deref, Range},
    sync::Arc,
};

use crate::{
    command::{
        recorder::CommandRecorder,
        sync::{AccessType, MemoryBarrier, PipelineBarrierConst},
    },
    resources::{
        alloc::{Allocator, BufferRequest, MemBuffer},
        buffer::vec_discrete::VecDiscrete,
    },
    Device,
};
use ash::extensions::khr;
use ash::vk;

pub struct AccelerationStructureLoader {
    device: Arc<Device>,
    loader: khr::AccelerationStructure,
}
impl AccelerationStructureLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::AccelerationStructure::new(device.instance(), &device);
        Self { device, loader }
    }
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl Deref for AccelerationStructureLoader {
    type Target = khr::AccelerationStructure;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}

// Acceleration structure can be resized and rebuilt
// Buffer data on CPU should be entirely optional.
// can update some entries of the AABB buffer without retransfering all datas.
// can append to the AABB buffer and resize AABB buffer. original data can do GPU to GPU transfers.
// can mark certain entries as DELETED, inactivate them so they don't show up in AS, and reuse them in the future when appending.
// Specialization for uniform memory.
pub struct AccelerationStructure {
    loader: Arc<AccelerationStructureLoader>,
    raw: vk::AccelerationStructureKHR,
    primitives_buffer: Option<VecDiscrete<vk::AabbPositionsKHR>>,
    backing_buffer: MemBuffer,
    compacted: bool,
}

// Builds many BLASs and at most one TLAS in batch
pub struct AccelerationStructureBuilder {
    loader: Arc<AccelerationStructureLoader>,
    allocator: Arc<Allocator>,
    builds: Vec<AccelerationStructureBuild>,
}

impl AccelerationStructureBuilder {
    pub fn add_aabb_blas(&mut self, item: AabbBlas) {
        self.builds.push(AccelerationStructureBuild::BlasAabb(item))
    }
    /// Must be called after all other BLAS builds
    pub fn add_tlas(&mut self, item: Tlas) {
        self.builds.push(AccelerationStructureBuild::Tlas(item))
    }
    // build on the device.
    pub fn build(&self, command_recorder: &mut CommandRecorder) -> Vec<AccelerationStructure> {
        // Calculate the total number of geometries
        let total_num_geometries = self
            .builds
            .iter()
            .map(AccelerationStructureBuild::num_geometries)
            .sum();
        let mut geometries: Vec<vk::AccelerationStructureGeometryKHR> =
            Vec::with_capacity(total_num_geometries);
        let mut build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> =
            Vec::with_capacity(total_num_geometries);
        let mut build_range_ptrs: Vec<*const vk::AccelerationStructureBuildRangeInfoKHR> =
            Vec::with_capacity(self.builds.len());

        let scratch_buffer_alignment = self
            .loader
            .device()
            .physical_device()
            .properties()
            .acceleration_structure
            .min_acceleration_structure_scratch_offset_alignment;

        // Write primitive data into a buffer.
        let all_primitives_size = self
            .builds
            .iter()
            .map(AccelerationStructureBuild::primitive_size)
            .sum::<usize>();
        let mut all_primitives_buffer = self
            .allocator
            .allocate_buffer(BufferRequest {
                size: all_primitives_size as u64,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_usage: gpu_alloc::UsageFlags::UPLOAD | gpu_alloc::UsageFlags::DEVICE_ADDRESS, // TODO: Is it faster to do this with a staging & device visible buffer?
                ..Default::default()
            })
            .unwrap();
        all_primitives_buffer.map_scoped(0, all_primitives_size, |mut buffer| {
            // write primitive buffers for all builds
            for build in self.builds.iter() {
                buffer = build.write_primitive_buffer(buffer);
            }
            assert_eq!(buffer.len(), 0);
        });

        // Create build infos
        let primitives_buffer_device_address = all_primitives_buffer.get_device_address();
        let mut current_primitives_buffer_device_address = primitives_buffer_device_address;
        let mut num_tlas: usize = 0;
        let mut num_blas: usize = 0;
        let (mut build_infos, build_sizes): (
            Vec<vk::AccelerationStructureBuildGeometryInfoKHR>,
            Vec<vk::AccelerationStructureBuildSizesInfoKHR>,
        ) = self
            .builds
            .iter()
            .map(|as_build| {
                let (ty, flags, geometry_range, geometry_primitive_counts) = match as_build {
                    AccelerationStructureBuild::BlasAabb(blas_builds) => {
                        let mut geometry_primitive_counts: Vec<u32> =
                            Vec::with_capacity(blas_builds.geometries.len());
                        let geometry_range: Range<usize> =
                            geometries.len()..(geometries.len() + blas_builds.geometries.len());
                        for (data, stride, primitive_count, geometry_flags) in
                            blas_builds.geometries.iter()
                        {
                            let geometry = vk::AccelerationStructureGeometryKHR {
                                geometry_type: vk::GeometryTypeKHR::AABBS,
                                geometry: vk::AccelerationStructureGeometryDataKHR {
                                    aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                                        data: vk::DeviceOrHostAddressConstKHR {
                                            device_address: {
                                                let d = current_primitives_buffer_device_address;
                                                current_primitives_buffer_device_address +=
                                                    data.len() as u64;
                                                d
                                            },
                                        },
                                        stride: *stride as u64,
                                        ..Default::default()
                                    },
                                },
                                flags: *geometry_flags,
                                ..Default::default()
                            };
                            num_blas += 1;
                            geometries.push(geometry);
                            geometry_primitive_counts.push(*primitive_count);
                            build_range_ptrs.push(unsafe {
                                // DANGEROUS: build_ranges can not be moved. So the initial capacity for build_ranges is important.
                                build_ranges.as_ptr().add(build_ranges.len())
                            });
                            build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                                primitive_count: *primitive_count,
                                primitive_offset: 0,
                                first_vertex: 0,
                                transform_offset: 0,
                            });
                        }

                        (
                            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                            blas_builds.flags,
                            geometry_range,
                            geometry_primitive_counts,
                        )
                    }
                    AccelerationStructureBuild::Tlas(tlas) => {
                        // Note: TLAS always only have one geometry.
                        let geometry = vk::AccelerationStructureGeometryKHR {
                            geometry_type: vk::GeometryTypeKHR::INSTANCES,
                            geometry: vk::AccelerationStructureGeometryDataKHR {
                                instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                                    array_of_pointers: vk::FALSE,
                                    data: vk::DeviceOrHostAddressConstKHR {
                                        device_address: {
                                            let d = current_primitives_buffer_device_address;
                                            let slice: &[vk::AccelerationStructureInstanceKHR] =
                                                &tlas.instances;
                                            current_primitives_buffer_device_address +=
                                                std::mem::size_of_val(slice) as u64;
                                            d
                                        },
                                    },
                                    ..Default::default()
                                },
                            },
                            flags: tlas.geometry_flags,
                            ..Default::default()
                        };
                        let geometry_range: Range<usize> = geometries.len()..(geometries.len() + 1);
                        num_tlas += 1;
                        geometries.push(geometry);
                        build_range_ptrs.push(unsafe {
                            // DANGEROUS: build_ranges can not be moved. So the initial capacity for build_ranges is important.
                            build_ranges.as_ptr().add(build_ranges.len())
                        });
                        build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count: tlas.instances.len() as u32,
                            primitive_offset: 0,
                            first_vertex: 0,
                            transform_offset: 0,
                        });
                        (
                            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                            tlas.flags,
                            geometry_range,
                            vec![tlas.instances.len() as u32],
                        )
                    }
                };
                let info = vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty,
                    flags,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    src_acceleration_structure: vk::AccelerationStructureKHR::null(),
                    dst_acceleration_structure: vk::AccelerationStructureKHR::null(),
                    geometry_count: geometry_range.len() as u32,
                    p_geometries: unsafe { geometries.as_ptr().add(geometry_range.start) },
                    pp_geometries: std::ptr::null(),
                    ..Default::default()
                };
                let mut build_size = unsafe {
                    self.loader.get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &info,
                        &geometry_primitive_counts,
                    )
                };
                // Pad to the scratch buffer alignment
                build_size.build_scratch_size = build_size
                    .build_scratch_size
                    .next_multiple_of(scratch_buffer_alignment as u64);
                drop(geometry_primitive_counts);
                (info, build_size)
            })
            .unzip();
        debug_assert_eq!(
            current_primitives_buffer_device_address - primitives_buffer_device_address,
            all_primitives_size as u64
        );

        // Create scratch buffers.
        // BLAS builds can overlap, so they need to use different regions of the scratch buffers.
        // TLAS builds always happen after all BLAS builds are finished, so it can reuse the scratch buffer.
        let (scratch_buffer_tlas_total, scratch_buffer_blas_total): (u64, u64) =
            build_sizes.iter().zip(build_infos.iter()).fold(
                (0_u64, 0_u64),
                |(tlas_total, blas_total), (build_size, build_info)| {
                    if build_info.ty == vk::AccelerationStructureTypeKHR::TOP_LEVEL {
                        (tlas_total + build_size.build_scratch_size, blas_total)
                    } else {
                        (tlas_total, blas_total + build_size.build_scratch_size)
                    }
                },
            );
        let scratch_buffer_total = scratch_buffer_blas_total.max(scratch_buffer_tlas_total);
        let scratch_buffer = self
            .allocator
            .allocate_buffer(BufferRequest {
                size: scratch_buffer_total,
                alignment: scratch_buffer_alignment as u64,
                // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03674
                // The buffer from which the buffer device address pInfos[i].scratchData.deviceAddress
                // is queried must have been created with VK_BUFFER_USAGE_STORAGE_BUFFER_BIT usage flag
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                ..Default::default()
            })
            .unwrap();
        let scratch_buffer_device_address = scratch_buffer.get_device_address();

        // Create acceleration structures
        let mut current_blas_scratch_buffer_device_address = scratch_buffer_device_address;
        let mut current_tlas_scratch_buffer_device_address = scratch_buffer_device_address;
        let acceleration_structures = build_infos
            .iter_mut()
            .zip(build_sizes.iter())
            .map(|(build_info, build_size)| unsafe {
                let backing_buffer = self
                    .allocator
                    .allocate_buffer(BufferRequest {
                        size: build_size.acceleration_structure_size,
                        alignment: 0,
                        usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                        memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        queue_families: &[],
                    })
                    .unwrap();
                let raw = self
                    .loader
                    .create_acceleration_structure(
                        &vk::AccelerationStructureCreateInfoKHR {
                            buffer: backing_buffer.buffer,
                            // We always use a dedicated buffer to store the acceleration structure so that when the
                            // acceleration structure is freed, we can easily destroy the buffer.
                            offset: 0,
                            size: build_size.acceleration_structure_size,
                            ty: build_info.ty,
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap();
                build_info.src_acceleration_structure = raw;

                let current_scratch_address =
                    if build_info.ty == vk::AccelerationStructureTypeKHR::TOP_LEVEL {
                        &mut current_tlas_scratch_buffer_device_address
                    } else {
                        &mut current_blas_scratch_buffer_device_address
                    };
                build_info.scratch_data = vk::DeviceOrHostAddressKHR {
                    device_address: *current_scratch_address,
                };
                *current_scratch_address += build_size.build_scratch_size;

                AccelerationStructure {
                    loader: self.loader.clone(),
                    raw,
                    primitives_buffer: None,
                    backing_buffer,
                    compacted: false,
                }
            })
            .collect::<Vec<_>>();

        // Actually record the build commands
        unsafe {
            assert_eq!(build_infos.len(), build_range_ptrs.len());
            let info_count = build_infos.len() as u32;
            assert_eq!(num_blas + num_tlas, info_count as usize);
            // First, build the BLASs
            self.loader.fp().cmd_build_acceleration_structures_khr(
                command_recorder.command_buffer,
                num_blas as u32,
                build_infos.as_ptr(),
                build_range_ptrs.as_ptr(),
            );
            // Pipeline barrier between the two builds.
            // This is necessarily because we assume taht the TLAS will contain some of the BLASs that we just built.
            command_recorder.simple_const_pipeline_barrier(&PipelineBarrierConst::new(
                Some(MemoryBarrier {
                    prev_accesses: &[AccessType::AccelerationStructureBuildWriteKHR],
                    next_accesses: &[AccessType::AccelerationStructureBuildReadKHR],
                }),
                &[],
                &[],
                vk::DependencyFlags::empty(),
            ));
            // Then, build the TLAS. INVARIANT: We assume that TLAS(s) are always placed after BLASs.
            self.loader.fp().cmd_build_acceleration_structures_khr(
                command_recorder.command_buffer,
                num_tlas as u32,
                build_infos.as_ptr().add(num_tlas),
                build_range_ptrs.as_ptr().add(num_tlas),
            );
        }
        acceleration_structures
    }
}

enum AccelerationStructureBuild {
    BlasAabb(AabbBlas),
    Tlas(Tlas),
}
impl AccelerationStructureBuild {
    fn num_geometries(&self) -> usize {
        match self {
            AccelerationStructureBuild::BlasAabb(build) => build.geometries.len(),
            AccelerationStructureBuild::Tlas(_) => 1,
        }
    }
    fn primitive_size(&self) -> usize {
        match self {
            AccelerationStructureBuild::BlasAabb(build) => build.num_primitives,
            AccelerationStructureBuild::Tlas(build) => build.instances.len(),
        }
    }
    /// Copy all geometry primitive buffers in this build into buf, returning the remaining buffer.
    fn write_primitive_buffer<'s, 'a>(&'s self, buf: &'a mut [u8]) -> &'a mut [u8] {
        fn fold_handler<'b>(buf: &'b mut [u8], geometry: &[u8]) -> &'b mut [u8] {
            buf[0..geometry.len()].copy_from_slice(geometry);
            &mut buf[geometry.len()..]
        }
        match self {
            AccelerationStructureBuild::BlasAabb(build) => build
                .geometries
                .iter()
                .map(|(data, _, _, _)| {
                    let slice: &[u8] = data;
                    slice
                })
                .fold(buf, fold_handler),
            AccelerationStructureBuild::Tlas(tlas) => unsafe {
                let slice: &[vk::AccelerationStructureInstanceKHR] = &tlas.instances;
                let size = std::mem::size_of_val(slice);
                let slice: &[u8] = std::slice::from_raw_parts(slice.as_ptr() as *const u8, size);
                fold_handler(buf, slice)
            },
        }
    }
}

/// Builds one AABB BLAS containing many geometries
pub struct AabbBlas {
    geometries: Vec<(Box<[u8]>, usize, u32, vk::GeometryFlagsKHR)>, // data, stride, num_primitives, flags
    flags: vk::BuildAccelerationStructureFlagsKHR,
    num_primitives: usize,
}

impl AabbBlas {
    /// T: The type of the interleaved data.
    pub fn add_geometry<T>(
        &mut self,
        primitives: Box<[(vk::AabbPositionsKHR, T)]>,
        flags: vk::GeometryFlagsKHR,
    ) {
        use std::alloc::Layout;
        // There might be two cases where vk::AabbPositionsKHR aren't layed out with a stride = 24
        // 1. The user wants to interleave some other metadata between vk::AabbPositionsKHR.
        //    Vulkan only guarantees that the intersection shader will be called for items within the AABB,
        //    so without raw f32 AABB data there might be visible artifacts.
        //    The primitive buffer likely needs to stay in device memory persistently for this, and the user might want to
        //    interleave some other metadata alongside the vk::AabbPositionsKHR.
        // 2. Using the same buffer for two or more geometries, interleaving the data. We assume that this use case
        //     would be very rare, so the design of the API does not consider this.
        let stride = {
            // verify that the layout is OK
            let layout = Layout::new::<(vk::AabbPositionsKHR, T)>();
            let padding_in_slice = layout.padding_needed_for(layout.align());
            // VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545: stride must be a multiple of 8
            let padding_in_buffer = layout.padding_needed_for(8);
            debug_assert_eq!(
                padding_in_slice, padding_in_buffer,
                "Type is incompatible. Stride between items must be a multiple of 8."
            );
            let stride = layout.size() + padding_in_buffer;
            debug_assert!(stride <= u32::MAX as usize);
            debug_assert!(stride % 8 == 0);
            stride
        };
        let num_primitives = primitives.len() as u32;
        self.num_primitives += num_primitives as usize;
        let primitives: Box<[u8]> = unsafe {
            let slice: &[(vk::AabbPositionsKHR, T)] = &primitives;
            let size = std::mem::size_of_val(slice);
            let data = Box::leak(primitives) as *mut _ as *mut u8;
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, size))
        };
        self.geometries
            .push((primitives, stride, num_primitives, flags));
    }
}

pub struct Tlas {
    pub instances: Box<[vk::AccelerationStructureInstanceKHR]>,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub geometry_flags: vk::GeometryFlagsKHR,
    // VkAccelerationStructureInstanceKHR are always tightly packed.
}
