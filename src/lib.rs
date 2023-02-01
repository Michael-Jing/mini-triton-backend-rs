#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod wrapper;

mod triton {
    mod backend {
        mod minimal {
            include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
            use core::ffi;
            // use cxx::{CxxString, CxxVector};
            use std::ffi::CString;

            use crate::wrapper::{Response, Request, self};

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
                    
                    let raw_request: *mut TRITONBACKEND_Request = unsafe {*requests.offset(r as isize) };
                    let request = Request::new(raw_request as *mut wrapper::TRITONBACKEND_Request).unwrap();

                    
                    let response = Response::new(request).unwrap();
                    let output = response.get_output("OUT0".to_owned(), TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32, vec![4]).unwrap();
                    let buffer = output.get_buffer(16, TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU , 0).unwrap();
                    buffer.write(vec![10, 9, 8, 7]).unwrap();
                    response.send(tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL, None );
                    
                }

                // call ResponseOutput first and then ResponseSend, OutputBuffer can be used to get a buffer to fill data in

                // other functions that could be useful, setstringparameter, setintparameter, setboolparameter

                return std::ptr::null_mut() as *mut TRITONSERVER_Error;
            }
        }
    }
}
