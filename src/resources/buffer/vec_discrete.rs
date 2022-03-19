use std::{ops::Range, sync::Arc};

use ash::vk;

use crate::resources::alloc::{Allocator, BufferRequest, MemBuffer};

use crate::command::recorder::CommandRecorder;

// A vector with a DEVICE_LOCAL memory and write-combined updates through staging.
pub struct VecDiscrete<T> {
    allocator: Arc<Allocator>,
    create_info: VecCreateInfo,
    buffer: Arc<MemBuffer>,
    len: u64,
    capacity: u64,

    // The staging buffer.
    ops: Vec<T>,
    // src_index, dst_index, num_items
    staging_to_device_copies: Vec<vk::BufferCopy>,
    free_ranges: Vec<Range<u64>>,
}

pub struct VecCreateInfo {
    pub flags: vk::BufferCreateFlags,
    pub usage: vk::BufferUsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub queue_family_indices: Box<[u32]>,
    pub alignment: u64,
    pub memory_usage: gpu_alloc::UsageFlags,
}

impl<T> VecDiscrete<T> {
    fn push_staging_to_device_copy(&mut self, src_index: usize, dst_index: u64, num_items: usize) {
        let s = std::mem::size_of::<T>() as u64;
        self.staging_to_device_copies.push(vk::BufferCopy {
            src_offset: s * src_index as u64,
            dst_offset: s * dst_index,
            size: s * num_items as u64,
        })
    }
    /// Add the specified item to the buffer.
    /// If pushing multiple items, please use `extend` instead of calling `push` multiple times.
    pub fn push(&mut self, item: T) {
        if let Some(range) = self.free_ranges.pop() {
            // use that range
            assert_ne!(range.start, range.end);
            self.update(item, range.start);
            if range.end - range.start > 1 {
                self.free_ranges.push(range.start + 1..range.end);
            }
        } else {
            // push.
            self.update(item, self.len);
            self.len += 1;
        }
    }
    /// Extends the buffer with the contents of an iterator.
    /// This will collect the contents of an iterator into a staging buffer.
    /// The current implementation try to fit this into existing empty slots range-by-range.
    /// This is likely faster than calling `push` multiple times. Batch your pushes as much as you can.
    pub fn extend(&mut self, items: impl IntoIterator<Item = T>) {
        let original_staging_len = self.ops.len();
        self.ops.extend(items);
        let num_items = self.ops.len() - original_staging_len;
        let mut num_items_left = num_items;
        let mut num_items_inserted: usize = 0;

        while num_items_left > 0 {
            if let Some(range) = self.free_ranges.pop() {
                let range_len = range.end - range.start;
                // Copy as much as we can
                let num_items_to_copy = num_items_left.min(range_len as usize);
                self.push_staging_to_device_copy(
                    original_staging_len + num_items_inserted,
                    range.start,
                    num_items_to_copy,
                );
                num_items_left -= num_items_to_copy;
                num_items_inserted += num_items_to_copy;
            } else {
                // No more free ranges. Insert all at tail.
                self.push_staging_to_device_copy(
                    original_staging_len + num_items_inserted,
                    self.len,
                    num_items_left,
                );
                self.len += num_items_left as u64;
                num_items_inserted += num_items_left;
                num_items_left = 0;
            }
        }
        assert_eq!(num_items_inserted, num_items);
    }

    /// Update the item at the specified index.
    /// If index is a freed or invalid index, item will be overwritten by a subsequent write.
    /// If updating multiple items, please use `update_range` instead of calling `update` multiple times.
    pub fn update(&mut self, item: T, index: u64) {
        self.update_range(std::iter::once(item), index);
    }

    /// If start_index..end_index contains freed or invalid indices, these entries can be overwritten by a subsequent write.
    /// This is likely faster than calling `update` multiple times. Batch your updates as much as you can.
    pub fn update_range(&mut self, items: impl IntoIterator<Item = T>, start_index: u64) {
        let original_staging_len = self.ops.len();
        self.ops.extend(items);
        let num_items = self.ops.len() - original_staging_len;
        self.push_staging_to_device_copy(original_staging_len, start_index, num_items);
    }
    /// Mark the item at the specified index as freed.
    pub fn free(&mut self, index: u64, length: u64) {
        self.free_ranges.push(index..(index + length));
    }

    /// command_pool: Recommands to be a command pool created with the transient flag.
    pub fn flush(&mut self, r: &mut CommandRecorder) {
        if self.len > self.capacity {
            // Needs to extend the buffer.
            let new_len = self.len + self.len / 2; // Inflate by 1.5x

            let new_buffer = self
                .allocator
                .allocate_buffer(BufferRequest {
                    size: new_len * std::mem::size_of::<T>() as u64,
                    alignment: self.create_info.alignment,
                    usage: self.create_info.usage | vk::BufferUsageFlags::TRANSFER_DST,
                    memory_usage: self.create_info.memory_usage,
                    sharing_mode: self.create_info.sharing_mode,
                    queue_families: &self.create_info.queue_family_indices,
                })
                .unwrap();
            let new_buffer = Arc::new(new_buffer);
            let old_buffer = std::mem::replace(&mut self.buffer, new_buffer.clone());
            // Perform device-to-device transfer.
            r.copy_buffer(
                old_buffer,
                new_buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: self.capacity,
                }],
            );
            self.capacity = new_len;
        }
        // now, drain pending changes.
        let mut staging_buffer = self
            .allocator
            .allocate_buffer(BufferRequest {
                size: std::mem::size_of::<T>() as u64 * self.ops.len() as u64,
                alignment: 0,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_usage: gpu_alloc::UsageFlags::UPLOAD,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_families: &[],
            })
            .unwrap();

        unsafe {
            let bytes = std::slice::from_raw_parts(
                self.ops.as_ptr() as *const u8,
                self.ops.len() * std::mem::size_of::<T>(),
            );
            staging_buffer.write_bytes(0, bytes);
            self.ops.clear()
        }
        r.copy_buffer(
            staging_buffer,
            self.buffer.clone(),
            &self.staging_to_device_copies,
        );
        self.staging_to_device_copies.clear();
    }
}

impl<T: Default> VecDiscrete<T> {
    /// Mark the item at the specified index as freed, and set the corresponding entry in
    /// the device buffer to Default.
    pub fn remove(&mut self, index: u64, length: u64) {
        self.free(index, length);
        self.update_range(std::iter::repeat_with(T::default), index);
    }
}
