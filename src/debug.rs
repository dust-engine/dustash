use ash::extensions::ext;
use ash::{prelude::VkResult, vk};
use std::ffi::{CStr, CString};

pub struct DebugUtilsMessenger {
    pub(crate) debug_utils: ext::DebugUtils,
    pub(crate) messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugUtilsMessenger {
    pub fn new(entry: &ash::Entry, instance: &mut ash::Instance) -> VkResult<Self> {
        let debug_utils = ext::DebugUtils::new(entry, instance);

        let messenger = unsafe {
            // Safety:
            // The application must ensure that vkCreateDebugUtilsMessengerEXT is not executed in parallel
            // with any Vulkan command that is also called with instance or child of instance as the dispatchable argument.
            // We do this by taking a mutable reference to Instance.
            debug_utils.create_debug_utils_messenger(
                &vk::DebugUtilsMessengerCreateInfoEXT {
                    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    pfn_user_callback: Some(debug_utils_callback),
                    //p_user_data: *mut c_void,
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self {
            debug_utils,
            messenger,
        })
    }
}

unsafe extern "system" fn debug_utils_callback<'a>(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    use tracing::Level;

    let callback_data: &'a vk::DebugUtilsMessengerCallbackDataEXT = &*callback_data;
    // We need to copy those strings here because the memory backing those
    // strings might get deallocated after this function returns.
    let message_id_name: &'a CStr = CStr::from_ptr(callback_data.p_message_id_name);
    let message_id_number = callback_data.message_id_number;
    let message: &'a CStr = CStr::from_ptr(callback_data.p_message);
    let _queue_labels: &'a [vk::DebugUtilsLabelEXT] = std::slice::from_raw_parts(
        callback_data.p_queue_labels,
        callback_data.queue_label_count as usize,
    );
    let _cmd_buf_labels: &'a [vk::DebugUtilsLabelEXT] = std::slice::from_raw_parts(
        callback_data.p_cmd_buf_labels,
        callback_data.cmd_buf_label_count as usize,
    );
    let _objects: &'a [vk::DebugUtilsObjectNameInfoEXT] =
        std::slice::from_raw_parts(callback_data.p_objects, callback_data.object_count as usize);

    let level = match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::DEBUG,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::INFO,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::WARN,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::ERROR,
        _ => Level::TRACE,
    };

    match level {
        Level::ERROR => {
            tracing::error!(message=?message_id_name, id=message_id_number, detail=?message)
        }
        Level::WARN => {
            tracing::warn!(message=?message_id_name, id=message_id_number, detail=?message)
        }
        Level::DEBUG => {
            tracing::debug!(message=?message_id_name, id=message_id_number, detail=?message)
        }
        Level::TRACE => {
            tracing::trace!(message=?message_id_name, id=message_id_number, detail=?message)
        }
        Level::INFO => {
            tracing::info!(message=?message_id_name, id=message_id_number, detail=?message)
        }
    };

    // The callback returns a VkBool32, which is interpreted in a layer-specified manner.
    // The application should always return VK_FALSE. The VK_TRUE value is reserved for use in layer development.
    vk::FALSE
}

/// Vulkan Object that can be associated with a name and/or a tag.
pub trait DebugObject: crate::HasDevice {
    fn object_handle(&mut self) -> u64;
    const OBJECT_TYPE: vk::ObjectType;
    fn set_name_cstr(&mut self, cstr: &CStr) -> VkResult<()> {
        unsafe {
            let raw_device = self.device().handle();
            let object_handle = self.object_handle();
            self.device()
                .instance()
                .debug_utils()
                .debug_utils
                .debug_utils_set_object_name(
                    raw_device,
                    &vk::DebugUtilsObjectNameInfoEXT {
                        object_type: Self::OBJECT_TYPE,
                        object_handle,
                        p_object_name: cstr.as_ptr(),
                        ..Default::default()
                    },
                )?;
        }
        Ok(())
    }
    fn set_name(&mut self, name: &str) -> VkResult<()> {
        let cstr = CString::new(name).expect("Name cannot contain null bytes");
        self.set_name_cstr(cstr.as_c_str())?;
        Ok(())
    }
    fn remove_name(&mut self) {
        unsafe {
            let raw_device = self.device().handle();
            let object_handle = self.object_handle();
            self.device()
                .instance()
                .debug_utils()
                .debug_utils
                .debug_utils_set_object_name(
                    raw_device,
                    &vk::DebugUtilsObjectNameInfoEXT {
                        object_type: Self::OBJECT_TYPE,
                        object_handle,
                        p_object_name: std::ptr::null(),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
}
