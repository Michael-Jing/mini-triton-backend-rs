#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use triton_backend_sys::sys::*;
use triton_backend_sys::wrapper::*;

use core::ffi;
// use cxx::{CxxString, CxxVector};
use std::ffi::CString;

// use crate::wrapper::{Response, Request, self};

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
        let raw_request: *mut TRITONBACKEND_Request = unsafe { *requests.offset(r as isize) };
        let request = Request::new(raw_request as *mut TRITONBACKEND_Request).unwrap();
        // deal with Request
        /*
        1. get_input_count
        2. get_input_by_index or get input by name
        3. get_input_properties
        4. get_buffer
        ? How to wrap read data from buffer
        */
        let input_count = request.get_input_count().unwrap();
        println!("input count is {:?}", input_count);
        let mut data: Vec<f32> = vec![];
        for i in 0..input_count {
            let input = request.get_input_by_index(i).unwrap();

            let input_properties = input.get_input_properties().unwrap();
            println!("input properties is {:?}", input_properties);
            let buffer_count = input_properties.buffer_count;
            let datatype = input_properties.datatype;
            for j in 0..buffer_count {
                let input_buffer = input.get_buffer(j, 0, 0, 0 as i32).unwrap();
                for n in input_buffer.into_iter() {
                    data.push(n as f32 * 2.0);
                    println!("n is {:?}", n);
                }
            }
        }

        let response = Response::new(&request).unwrap();
        let output = response
            .get_output(
                "OUT0".to_owned(),
                TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
                vec![4],
            )
            .unwrap();
        let buffer = output
            .get_buffer(
                16,
                TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                0,
                TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
            )
            .unwrap();
        //let data: Vec<f32> = vec![10., 9., 8., 1.];
        buffer.write(data).unwrap(); // datatype should match the expected datatype, otherwise the write may result in a slient fail or segmentation fault

        response.send(
            tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            None,
        );
    }

    // call ResponseOutput first and then ResponseSend, OutputBuffer can be used to get a buffer to fill data in

    // other functions that could be useful, setstringparameter, setintparameter, setboolparameter

    return std::ptr::null_mut() as *mut TRITONSERVER_Error;
}
