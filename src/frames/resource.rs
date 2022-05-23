use super::AcquiredFrame;

pub struct PerFrame<T, const PER_IMAGE: bool = false> {
    resources: Vec<Option<T>>,
}

impl<T, const PER_IMAGE: bool> Default for PerFrame<T, PER_IMAGE> {
    fn default() -> Self {
        Self {
            resources: Vec::new(),
        }
    }
}

impl<T, const PER_IMAGE: bool> PerFrame<T, PER_IMAGE> {
    pub fn get(&self, frame: &AcquiredFrame) -> Option<&T> {
        if frame.invalidate_images {
            return None;
        }
        let index = if PER_IMAGE {
            frame.image_index as usize
        } else {
            frame.frame_index
        };
        self.resources.get(index).map_or(None, Option::as_ref)
    }
    pub fn get_mut(&mut self, frame: &AcquiredFrame) -> Option<&mut T> {
        if frame.invalidate_images {
            return None;
        }
        let index = if PER_IMAGE {
            frame.image_index as usize
        } else {
            frame.frame_index
        };
        self.resources.get_mut(index).map_or(None, Option::as_mut)
    }
    pub fn get_or_else(
        &mut self,
        frame: &AcquiredFrame,
        force_update: impl FnOnce(&T) -> bool,
        f: impl FnOnce(Option<T>) -> T,
    ) -> &mut T {
        let index = if PER_IMAGE {
            frame.image_index as usize
        } else {
            frame.frame_index
        };
        if self.resources.len() <= index {
            self.resources.resize_with(index + 1, || None);
        }
        let slot = &mut self.resources[index];
        let original = slot.take();
        if frame.invalidate_images || original.is_none() || force_update(original.as_ref().unwrap())
        {
            // Drop first, then create new
            *slot = Some(f(original));
        } else {
            *slot = original;
        }
        slot.as_mut().unwrap()
    }
}
