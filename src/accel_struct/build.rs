use std::{mem::ManuallyDrop, ops::Range, sync::Arc};

use super::AccelerationStructure;
use super::AccelerationStructureLoader;
use crate::{
    command::{
        recorder::CommandRecorder,
        sync::{AccessType, MemoryBarrier, PipelineBarrierConst},
    },
    resources::alloc::{Allocator, BufferRequest},
};

use ash::vk;

// Builds many BLASs and at most one TLAS in batch
pub struct AccelerationStructureBuilder {
    loader: Arc<AccelerationStructureLoader>,
    allocator: Arc<Allocator>,
    builds: Vec<AccelerationStructureBuild>,
}

impl AccelerationStructureBuilder {
    pub fn add_aabb_blas(&mut self, item: AabbBlasBuilder) {
        self.builds
            .push(item.build(self.loader.clone(), self.allocator.clone()))
    }
    /// Must be called after all other BLAS builds
    pub fn add_tlas(
        &mut self,
        instances: Box<[vk::AccelerationStructureInstanceKHR]>,
        geometry_flags: vk::GeometryFlagsKHR,
        tlas_flags: vk::BuildAccelerationStructureFlagsKHR,
    ) {
        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                    array_of_pointers: vk::FALSE,
                    ..Default::default()
                },
            },
            flags: geometry_flags,
            ..Default::default()
        };
        let mut build_size = unsafe {
            self.loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                    flags: tlas_flags,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    geometry_count: 1,
                    p_geometries: &geometry,
                    ..Default::default()
                },
                &[instances.len() as u32],
            )
        };

        let scratch_buffer_alignment = self
            .loader
            .device()
            .physical_device()
            .properties()
            .acceleration_structure
            .min_acceleration_structure_scratch_offset_alignment;
        build_size.build_scratch_size = build_size
            .build_scratch_size
            .next_multiple_of(scratch_buffer_alignment as u64);
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
        let raw = unsafe {
            self.loader
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR {
                        buffer: backing_buffer.buffer,
                        offset: 0,
                        size: build_size.acceleration_structure_size,
                        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        let accel_struct = AccelerationStructure {
            loader: self.loader.clone(),
            raw,
            primitives_buffer: None,
            backing_buffer: ManuallyDrop::new(backing_buffer),
            compacted: false,
            flags: tlas_flags,
            num_primitives: instances.len() as u64,
            geometries_num_primitives: vec![instances.len() as u32],
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        };
        self.builds.push(AccelerationStructureBuild {
            accel_struct,
            build_size,
            ty: AccelerationStructureBuildType::Tlas {
                instances,
                geometry_flags,
            },
        })
    }
    // build on the device.
    pub fn build(self, command_recorder: &mut CommandRecorder) -> Vec<Arc<AccelerationStructure>> {
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
        // Create scratch buffers.
        // BLAS builds can overlap, so they need to use different regions of the scratch buffers.
        // TLAS builds always happen after all BLAS builds are finished, so it can reuse the scratch buffer.
        let (scratch_buffer_tlas_total, scratch_buffer_blas_total): (u64, u64) = self
            .builds
            .iter()
            .fold((0_u64, 0_u64), |(tlas_total, blas_total), build| {
                if build.accel_struct.ty == vk::AccelerationStructureTypeKHR::TOP_LEVEL {
                    (tlas_total + build.build_size.build_scratch_size, blas_total)
                } else {
                    (tlas_total, blas_total + build.build_size.build_scratch_size)
                }
            });
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

        // Write primitive data into a buffer.
        let all_primitives_size = self
            .builds
            .iter()
            .map(|build| build.accel_struct.num_primitives as usize)
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
        let mut current_blas_scratch_buffer_device_address = scratch_buffer_device_address;
        let mut current_tlas_scratch_buffer_device_address = scratch_buffer_device_address;

        let build_infos = self
            .builds
            .iter()
            .map(|as_build| {
                build_range_ptrs.push(unsafe {
                    // DANGEROUS: build_ranges can not be moved. So the initial capacity for build_ranges is important.
                    build_ranges.as_ptr().add(build_ranges.len())
                });

                // Add geometries
                let geometry_range = match &as_build.ty {
                    AccelerationStructureBuildType::BlasAabb(blas_builds) => {
                        build_ranges.extend(
                            as_build
                                .accel_struct
                                .geometries_num_primitives
                                .iter()
                                .map(|a| vk::AccelerationStructureBuildRangeInfoKHR {
                                    primitive_count: *a,
                                    primitive_offset: 0,
                                    first_vertex: 0,
                                    transform_offset: 0,
                                }),
                        );

                        let geometry_range: Range<usize> =
                            geometries.len()..(geometries.len() + blas_builds.geometries.len());
                        // Insert geometries
                        geometries.extend(blas_builds.to_geometry_infos().into_iter().map(
                            |mut geometry_info: vk::AccelerationStructureGeometryKHR| unsafe {
                                let data_len = geometry_info.geometry.aabbs.data.device_address;
                                geometry_info.geometry.aabbs.data.device_address =
                                    current_primitives_buffer_device_address;
                                current_primitives_buffer_device_address += data_len;
                                geometry_info
                            },
                        )); // TODO: address
                        geometry_range
                    }
                    AccelerationStructureBuildType::Tlas {
                        instances,
                        geometry_flags,
                    } => {
                        build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count: instances.len() as u32,
                            primitive_offset: 0,
                            first_vertex: 0,
                            transform_offset: 0,
                        });
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
                                                &instances;
                                            current_primitives_buffer_device_address +=
                                                std::mem::size_of_val(slice) as u64;
                                            d
                                        },
                                    },
                                    ..Default::default()
                                },
                            },
                            flags: *geometry_flags,
                            ..Default::default()
                        };
                        let geometry_range: Range<usize> = geometries.len()..(geometries.len() + 1);
                        geometries.push(geometry);
                        geometry_range
                    }
                };
                let info = vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty: as_build.accel_struct.ty,
                    flags: as_build.accel_struct.flags,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    src_acceleration_structure: as_build.accel_struct.raw,
                    dst_acceleration_structure: vk::AccelerationStructureKHR::null(),
                    geometry_count: geometry_range.len() as u32,
                    p_geometries: unsafe { geometries.as_ptr().add(geometry_range.start) },
                    scratch_data: {
                        let current_scratch_address = if as_build.accel_struct.ty
                            == vk::AccelerationStructureTypeKHR::TOP_LEVEL
                        {
                            &mut current_tlas_scratch_buffer_device_address
                        } else {
                            &mut current_blas_scratch_buffer_device_address
                        };
                        let d = vk::DeviceOrHostAddressKHR {
                            device_address: *current_scratch_address,
                        };
                        *current_scratch_address += as_build.build_size.build_scratch_size;
                        d
                    },
                    ..Default::default()
                };
                info
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(
            current_primitives_buffer_device_address - primitives_buffer_device_address,
            all_primitives_size as u64
        );

        // Actually record the build commands
        let (num_tlas, num_blas): (u32, u32) =
            self.builds
                .iter()
                .fold((0_u32, 0_u32), |(tlas_total, blas_total), build| {
                    if build.accel_struct.ty == vk::AccelerationStructureTypeKHR::TOP_LEVEL {
                        (tlas_total + 1, blas_total)
                    } else {
                        (tlas_total, blas_total + 1)
                    }
                });
        unsafe {
            assert_eq!(build_infos.len(), build_range_ptrs.len());
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
                build_infos.as_ptr().add(num_tlas as usize),
                build_range_ptrs.as_ptr().add(num_tlas as usize),
            );
        }

        // Finally, add the dependency data
        let acceleration_structures = self
            .builds
            .into_iter()
            .map(|build| Arc::new(build.accel_struct))
            .collect::<Vec<_>>();
        command_recorder.referenced_resources.extend(
            acceleration_structures
                .iter()
                .cloned()
                .map(|a| Box::new(a) as Box<dyn Send>),
        );
        command_recorder
            .referenced_resources
            .push(Box::new(scratch_buffer));
        command_recorder
            .referenced_resources
            .push(Box::new(all_primitives_buffer));
        acceleration_structures
    }
}

struct AccelerationStructureBuild {
    accel_struct: AccelerationStructure,
    build_size: vk::AccelerationStructureBuildSizesInfoKHR,
    ty: AccelerationStructureBuildType,
}
enum AccelerationStructureBuildType {
    BlasAabb(AabbBlas),
    Tlas {
        instances: Box<[vk::AccelerationStructureInstanceKHR]>,
        geometry_flags: vk::GeometryFlagsKHR,
    },
}
impl AccelerationStructureBuild {
    // Total number of geometries
    fn num_geometries(&self) -> usize {
        match &self.ty {
            AccelerationStructureBuildType::BlasAabb(build) => build.geometries.len(),
            AccelerationStructureBuildType::Tlas { .. } => 1,
        }
    }
    /// Copy all geometry primitive buffers in this build into buf, returning the remaining buffer.
    fn write_primitive_buffer<'s, 'a>(&'s self, buf: &'a mut [u8]) -> &'a mut [u8] {
        fn fold_handler<'b>(buf: &'b mut [u8], geometry: &[u8]) -> &'b mut [u8] {
            buf[0..geometry.len()].copy_from_slice(geometry);
            &mut buf[geometry.len()..]
        }
        match &self.ty {
            AccelerationStructureBuildType::BlasAabb(build) => build
                .geometries
                .iter()
                .map(|(data, _, _)| {
                    let slice: &[u8] = data;
                    slice
                })
                .fold(buf, fold_handler),
            AccelerationStructureBuildType::Tlas { instances, .. } => unsafe {
                let slice: &[vk::AccelerationStructureInstanceKHR] = &instances;
                let size = std::mem::size_of_val(slice);
                let slice: &[u8] = std::slice::from_raw_parts(slice.as_ptr() as *const u8, size);
                fold_handler(buf, slice)
            },
        }
    }
}

/// Builds one AABB BLAS containing many geometries
pub struct AabbBlasBuilder {
    geometries: Vec<(Box<[u8]>, usize, vk::GeometryFlagsKHR)>, // data, stride, num_primitives, flags
    flags: vk::BuildAccelerationStructureFlagsKHR,
    num_primitives: u64,
    geometry_primitive_counts: Vec<u32>,
}
pub struct AabbBlas {
    geometries: Box<[(Box<[u8]>, usize, vk::GeometryFlagsKHR)]>, // data, stride, num_primitives, flags
}

impl AabbBlas {
    fn to_geometry_infos<'a>(
        &'a self,
    ) -> impl IntoIterator<Item = vk::AccelerationStructureGeometryKHR> + 'a {
        self.geometries.iter().map(
            |(data, stride, flags)| vk::AccelerationStructureGeometryKHR {
                geometry_type: vk::GeometryTypeKHR::AABBS,
                geometry: vk::AccelerationStructureGeometryDataKHR {
                    aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                        data: vk::DeviceOrHostAddressConstKHR {
                            device_address: data.len() as u64,
                        },
                        stride: *stride as u64,
                        ..Default::default()
                    },
                },
                flags: *flags,
                ..Default::default()
            },
        )
    }
}

impl AabbBlasBuilder {
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
        self.num_primitives += num_primitives as u64;
        let primitives: Box<[u8]> = unsafe {
            let slice: &[(vk::AabbPositionsKHR, T)] = &primitives;
            let size = std::mem::size_of_val(slice);
            let data = Box::leak(primitives) as *mut _ as *mut u8;
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, size))
        };
        self.geometries.push((primitives, stride, flags));
        self.geometry_primitive_counts.push(num_primitives);
    }
    fn build(
        self,
        loader: Arc<AccelerationStructureLoader>,
        allocator: Arc<Allocator>,
    ) -> AccelerationStructureBuild {
        let geometries: Vec<vk::AccelerationStructureGeometryKHR> = self
            .geometries
            .iter()
            .map(|(_, stride, flags)| vk::AccelerationStructureGeometryKHR {
                geometry_type: vk::GeometryTypeKHR::AABBS,
                geometry: vk::AccelerationStructureGeometryDataKHR {
                    aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                        stride: *stride as u64,
                        ..Default::default()
                    },
                },
                flags: *flags,
                ..Default::default()
            })
            .collect();
        unsafe {
            let mut build_size = loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    flags: self.flags,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    src_acceleration_structure: vk::AccelerationStructureKHR::null(),
                    dst_acceleration_structure: vk::AccelerationStructureKHR::null(),
                    geometry_count: self.geometries.len() as u32,
                    p_geometries: geometries.as_ptr(),
                    pp_geometries: std::ptr::null(),
                    ..Default::default()
                },
                &self.geometry_primitive_counts,
            );
            let scratch_buffer_alignment = loader
                .device()
                .physical_device()
                .properties()
                .acceleration_structure
                .min_acceleration_structure_scratch_offset_alignment;
            build_size.build_scratch_size = build_size
                .build_scratch_size
                .next_multiple_of(scratch_buffer_alignment as u64);
            let backing_buffer = allocator
                .allocate_buffer(BufferRequest {
                    size: build_size.acceleration_structure_size,
                    alignment: 0,
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                    memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    queue_families: &[],
                })
                .unwrap();
            let raw = loader
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR {
                        buffer: backing_buffer.buffer,
                        offset: 0,
                        size: build_size.acceleration_structure_size,
                        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let accel_struct = AccelerationStructure {
                loader,
                raw,
                primitives_buffer: None,
                backing_buffer: ManuallyDrop::new(backing_buffer),
                compacted: false,
                flags: self.flags,
                num_primitives: self.num_primitives,
                geometries_num_primitives: self.geometry_primitive_counts,
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            };
            AccelerationStructureBuild {
                accel_struct,
                build_size,
                ty: AccelerationStructureBuildType::BlasAabb(AabbBlas {
                    geometries: self.geometries.into_boxed_slice(),
                }),
            }
        }
    }
}
