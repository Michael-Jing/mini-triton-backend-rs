#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod triton {
    mod backend {
        mod exp_mini_backend {
            include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
            use core::ffi;
            // use cxx::{CxxString, CxxVector};
            use std::ffi::CString;

            macro_rules! RETURN_ERROR_IF_TRUE {
                ($P: expr, $C: expr, $M: expr) => {
                    if $P {
                        return TRITONSERVER_ErrorNew($C, std::ffi::CString::new($M));
                    }
                };
            }

            macro_rules! RETURN_IF_ERROR {
                ($X: expr) => {
                    let rie_err__: *mut TRITONSERVER_Error = $X;
                    if !rie_err__.is_null() {
                        return rie_err__;
                    }
                };
            }

            #[no_mangle]
            pub extern "C" fn TRITONBACKEND_ModelInstanceExecute(
                instance: *mut TRITONBACKEND_ModelInstance,
                requests: *mut *mut TRITONBACKEND_Request,
                request_count: u32,
            ) -> *mut TRITONSERVER_Error {
                // usable functions TRITONBACKEND_ResponseNew
                for r in 0..request_count {
                    let mut response: *mut TRITONBACKEND_Response =
                        std::ptr::null_mut() as *mut TRITONBACKEND_Response;
                    unsafe {
                        let request: *mut TRITONBACKEND_Request = *requests.offset(r as isize); // as *mut ffi::TRITONBACKEND_Request;
                        RETURN_IF_ERROR!(TRITONBACKEND_ResponseNew(&mut response, request));
                    }
                    let mut output: Box<*mut TRITONBACKEND_Output> =
                        Box::new(std::ptr::null_mut() as *mut TRITONBACKEND_Output);
                    let mut output = Box::into_raw(output);
                    let nameRaw = CString::new("OUT0").unwrap();
                    let name: *const std::ffi::c_char = nameRaw.as_ptr();
                    let datatype: TRITONSERVER_DataType =
                        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32;
                    let shape: i64 = 4;
                    let dims_count = 1;
                    unsafe {
                        TRITONBACKEND_ResponseOutput(
                            response, output, name, datatype, &shape, dims_count,
                        );
                    }
                    let mut buffer: Box<*mut ffi::c_void> =
                        Box::new(std::ptr::null_mut() as *mut ffi::c_void);
                    let mut buffer = Box::into_raw(buffer);
                    let buffer_byte_size = 16;
                    let mut memory_type = TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
                    let mut memory_type_id: i64 = 0;
                    unsafe {
                        RETURN_IF_ERROR!(TRITONBACKEND_OutputBuffer(
                            *output,
                            buffer,
                            buffer_byte_size,
                            &mut memory_type as *mut u32,
                            &mut memory_type_id as *mut i64,
                        ));
                    }
                    let mut buffer: *mut *mut f32 = buffer as *mut *mut f32;
                    unsafe {
                        let mut begin = *buffer;
                        for i in 0..4 {
                            *begin.offset(i) = (i * 4) as f32;
                        }
                    }
                    unsafe {
                        // notes for safe wrap: TRITONBACKEND_ResponseSend  takes ownership of response;
                        TRITONBACKEND_ResponseSend(
                        response,
                        tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                        std::ptr::null_mut() as *mut TRITONSERVER_Error,
                    );
                    }
                }

                // call ResponseOutput first and then ResponseSend, OutputBuffer can be used to get a buffer to fill data in

                // other functions that could be useful, setstringparameter, setintparameter, setboolparameter

                return std::ptr::null_mut() as *mut TRITONSERVER_Error;
            }
        }
    }
}
