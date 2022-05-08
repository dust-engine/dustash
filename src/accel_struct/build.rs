use std::{ops::Range, sync::Arc};

use super::AccelerationStructure;
use super::AccelerationStructureLoader;
use crate::resources::alloc::MemBuffer;
use crate::resources::alloc::{Allocator, BufferRequest};
use crate::sync::CommandsFuture;

use crate::HasDevice;
use ash::vk;

/// Builds many acceleration structures in batch.
pub struct AccelerationStructureBuilder {
    loader: Arc<AccelerationStructureLoader>,
    allocator: Arc<Allocator>,
    builds: Vec<AccelerationStructureBuild>,
}

impl AccelerationStructureBuilder {
    pub fn new(loader: Arc<AccelerationStructureLoader>, allocator: Arc<Allocator>) -> Self {
        Self {
            loader,
            allocator,
            builds: Vec::new(),
        }
    }
    pub fn add_aabb_blas(&mut self, item: AabbBlasBuilder) {
        self.builds
            .push(item.build(self.loader.clone(), self.allocator.clone()))
    }
    // build on the device.
    /// Instead of calling VkCmdBuildAccelerationStructure multiple times, it calls VkCmdBuildAccelerationStructure
    /// in batch mode, once for BLAS and once for TLAS, with a pipeline barrier inbetween.  
    pub fn build(self, commands_future: &mut CommandsFuture) -> Vec<Arc<AccelerationStructure>> {
        // Calculate the total number of geometries
        let total_num_geometries = self.builds.iter().map(|build| build.geometries.len()).sum();
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
        let scratch_buffer_total: u64 = self
            .builds
            .iter()
            .map(|build| build.build_size.build_scratch_size)
            .sum();
        let scratch_buffer = self
            .allocator
            .allocate_buffer(BufferRequest {
                size: scratch_buffer_total,
                alignment: scratch_buffer_alignment as u64,
                // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03674
                // The buffer from which the buffer device address pInfos[i].scratchData.deviceAddress
                // is queried must have been created with VK_BUFFER_USAGE_STORAGE_BUFFER_BIT usage flag
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                    | gpu_alloc::UsageFlags::DEVICE_ADDRESS,
                ..Default::default()
            })
            .unwrap();
        let scratch_buffer_device_address = scratch_buffer.get_device_address();

        // Create build infos
        let mut current_scratch_buffer_device_address = scratch_buffer_device_address;

        let build_infos = self
            .builds
            .iter()
            .map(|as_build| {
                build_range_ptrs.push(unsafe {
                    // DANGEROUS: build_ranges can not be moved. So the initial capacity for build_ranges is important.
                    build_ranges.as_ptr().add(build_ranges.len())
                });

                // Add geometries
                build_ranges.extend(as_build.accel_struct.geometries_num_primitives.iter().map(
                    |a| vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: *a,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    },
                ));

                let geometry_range: Range<usize> =
                    geometries.len()..(geometries.len() + as_build.geometries.len());
                // Insert geometries
                geometries.extend(aabbs_to_geometry_infos(&as_build.geometries));
                vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    flags: as_build.accel_struct.flags,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    dst_acceleration_structure: as_build.accel_struct.raw,
                    geometry_count: as_build.geometries.len() as u32,
                    p_geometries: unsafe { geometries.as_ptr().add(geometry_range.start) },
                    scratch_data: {
                        let d = vk::DeviceOrHostAddressKHR {
                            device_address: current_scratch_buffer_device_address,
                        };
                        current_scratch_buffer_device_address +=
                            as_build.build_size.build_scratch_size;
                        d
                    },
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();

        // Actually record the build commands
        commands_future.then_commands(|recorder| {
            unsafe {
                assert_eq!(build_infos.len(), build_range_ptrs.len());
                // First, build the BLASs
                self.loader.fp().cmd_build_acceleration_structures_khr(
                    recorder.command_buffer,
                    build_infos.len() as u32,
                    build_infos.as_ptr(),
                    build_range_ptrs.as_ptr(),
                );
            }

            // Finally, add the dependency data
            recorder.referenced_resources.extend(
                self.builds
                    .iter()
                    .flat_map(|build| build.geometries.iter().map(|g| g.0.clone()))
                    .map(|arc| Box::new(arc) as Box<dyn Send + Sync>),
            );
            let acceleration_structures = self
                .builds
                .into_iter()
                .map(|build| Arc::new(build.accel_struct))
                .collect::<Vec<_>>();
            recorder.referenced_resources.extend(
                acceleration_structures
                    .iter()
                    .cloned()
                    .map(|a| Box::new(a) as Box<dyn Send + Sync>),
            );
            recorder.referenced_resources.push(Box::new(scratch_buffer));
            acceleration_structures
        })
    }
}

struct AccelerationStructureBuild {
    accel_struct: AccelerationStructure,
    build_size: vk::AccelerationStructureBuildSizesInfoKHR,
    geometries: Box<[(Arc<MemBuffer>, usize, vk::GeometryFlagsKHR)]>, // data, stride, flags
    primitive_datasize: usize,
}

/// Builds one AABB BLAS containing many geometries
pub struct AabbBlasBuilder {
    geometries: Vec<(Arc<MemBuffer>, usize, vk::GeometryFlagsKHR)>, // data, stride, num_primitives, flags
    flags: vk::BuildAccelerationStructureFlagsKHR,
    num_primitives: u64,
    geometry_primitive_counts: Vec<u32>,
    primitive_datasize: usize,
}

fn aabbs_to_geometry_infos(
    geometries: &[(Arc<MemBuffer>, usize, vk::GeometryFlagsKHR)],
) -> impl IntoIterator<Item = vk::AccelerationStructureGeometryKHR> + '_ {
    geometries.iter().map(
        |(data, stride, flags)| vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::AABBS,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                    data: vk::DeviceOrHostAddressConstKHR {
                        device_address: data.get_device_address(),
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

impl AabbBlasBuilder {
    pub fn new(flags: vk::BuildAccelerationStructureFlagsKHR) -> Self {
        Self {
            geometries: Vec::new(),
            flags,
            num_primitives: 0,
            geometry_primitive_counts: Vec::new(),
            primitive_datasize: 0,
        }
    }
    /// T: The type of the interleaved data.
    pub fn add_geometry<T>(&mut self, primitives: Arc<MemBuffer>, flags: vk::GeometryFlagsKHR) {
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
        let num_primitives = primitives.size() / stride as u64;
        self.num_primitives += num_primitives;
        self.primitive_datasize += primitives.size() as usize;
        self.geometries.push((primitives, stride, flags));
        self.geometry_primitive_counts.push(num_primitives as u32);
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
                        // No need to touch the data pointer here, since this VkAccelerationStructureGeometryKHR is
                        // used for VkgetAccelerationStructureBuildSizes only.
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
                    geometry_count: self.geometries.len() as u32,
                    p_geometries: geometries.as_ptr(),
                    ..Default::default()
                },
                &self.geometry_primitive_counts,
            );
            let scratch_buffer_alignment = loader
                .physical_device()
                .properties()
                .acceleration_structure
                .min_acceleration_structure_scratch_offset_alignment;
            build_size.build_scratch_size = build_size
                .build_scratch_size
                .next_multiple_of(scratch_buffer_alignment as u64);
            let mut accel_struct = AccelerationStructure::new(
                loader,
                &allocator,
                build_size.acceleration_structure_size,
                super::AccelerationStructureType::BottomLevelAABBs,
            );
            accel_struct.geometries_num_primitives = self.geometry_primitive_counts;
            accel_struct.flags = self.flags;
            AccelerationStructureBuild {
                accel_struct,
                build_size,
                geometries: self.geometries.into_boxed_slice(),
                primitive_datasize: self.primitive_datasize,
            }
        }
    }
}
