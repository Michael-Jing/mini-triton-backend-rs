use std::ffi::{CString, CStr};
use std::mem::MaybeUninit;
use std::marker::PhantomData;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub struct ServerError {
    _error: *mut TRITONSERVER_Error,
}
pub struct Input {
    _input: *mut TRITONBACKEND_Input,
}

pub struct InputProperties<'a> {
    name: Option<&'a str>,
    datatype: Option<TRITONSERVER_DataType>,
    shape: Option<&'a[i64]>,
    dims_count: Option<u32>,
    byte_size: Option<u64>,
    buffer_count: Option<u32>,
    phantom: PhantomData<&'a Input>,
}
pub struct Buffer {
    _buffer: *const std::os::raw::c_void,
    buffer_byte_size: u64,
    memory_type: TRITONSERVER_MemoryType,
    memory_type_id: i64,
}
impl Input {
    pub fn InputProperties(&self) -> Result<InputProperties, ServerError> {
        let mut name: *mut *const std::os::raw::c_char = std::ptr::null_mut() as *mut *const std::os::raw::c_char;
        let mut datatype : * mut TRITONSERVER_DataType = std::ptr::null_mut() as *mut TRITONSERVER_DataType;
        let mut shapeRaw : *mut *const i64 = std::ptr::null_mut() as *mut *const i64;
        let mut dims_count: *mut u32 = std::ptr::null_mut() as *mut u32;
        let mut byte_size : *mut u64 = std::ptr::null_mut() as *mut u64;
        let mut buffer_count : *mut u32 = std::ptr::null_mut() as *mut u32;
        let err = unsafe { TRITONBACKEND_InputProperties(
            self._input,
            name,
            datatype,
            shapeRaw,
            dims_count,
            byte_size,
            buffer_count,
        )};
        let dims_count = if dims_count.is_null() {
            None
        } else {
            Some(unsafe{*dims_count})
        };

        
       let shape = if shapeRaw.is_null() {
            None
        } else if let None = dims_count {
            None
        } else {
            Some(
                unsafe {std::slice::from_raw_parts(*shapeRaw, dims_count.unwrap() as usize)})
        };

        return if err.is_null() {
            Ok(InputProperties {
                // fill in the fields
                name: if name.is_null() {None} else {
                    unsafe {
                        Some(CStr::from_ptr(*name).to_str().unwrap_or(""))
                    } 
                },

                datatype: if datatype.is_null() {None} else {
                    unsafe {
                        Some(*datatype)
                    }
                },
                shape,
                dims_count,
                byte_size: if byte_size.is_null() {None} else {
                    unsafe {Some(*byte_size)}
                },
                buffer_count: if buffer_count.is_null() {None} else {
                    unsafe {Some(*buffer_count)}
                },
                phantom: PhantomData,
            })
        } else {
            Err(ServerError { _error: err })
        };
    }

    pub fn InputBuffer(&self, index: u32) -> Result<Buffer, ServerError> {
        /*
             TRITONBACKEND_InputBuffer(
            input: *mut TRITONBACKEND_Input,
            index: u32,
            buffer: *mut *const ::std::os::raw::c_void,
            buffer_byte_size: *mut u64,
            memory_type: *mut TRITONSERVER_MemoryType,
            memory_type_id: *mut i64,
        ) -> *mut TRITONSERVER_Error; */
        let mut buffer: *const std::os::raw::c_void =
            std::ptr::null() as *const std::os::raw::c_void;
        let mut buffer_byte_size: u64 = 0;
        let mut memory_type: TRITONSERVER_MemoryType = 0;
        let mut memory_type_id: i64 = 0;
        let err = unsafe{ TRITONBACKEND_InputBuffer(
            self._input,
            index,
            &mut buffer,
            &mut buffer_byte_size,
            &mut memory_type,
            &mut memory_type_id,
        )};
        return if err.is_null() {
            Ok(Buffer {
                _buffer: buffer,
                buffer_byte_size,
                memory_type,
                memory_type_id,
            })
        } else {
            Err(ServerError { _error: err })
        };
    }
}

pub struct Request {
    _request: *mut TRITONBACKEND_Request,
}
impl Request {
    pub fn InputCount(&self) -> Result<u32, ServerError> {
        let mut count = 0;
        let err = unsafe{ TRITONBACKEND_RequestInputCount(self._request, &mut count)};
        return if err.is_null() {
             Ok(count)
        } else {
            Err(ServerError { _error: err })
        }
    }

    pub fn InputByIndex(&self, index: u32) -> Result<Input, ServerError> {
        /*
             TRITONBACKEND_RequestInputByIndex(
            request: *mut TRITONBACKEND_Request,
            index: u32,
            input: *mut *mut TRITONBACKEND_Input,
        ) -> *mut TRITONSERVER_Error;
         */
        let mut input: *mut TRITONBACKEND_Input = std::ptr::null_mut() as *mut TRITONBACKEND_Input;
        let err = unsafe {TRITONBACKEND_RequestInputByIndex(self._request, index, &mut input)};
        return if err.is_null() {
            Ok(Input { _input: input })
        } else {
            Err(ServerError { _error: err })
        };
    }
}

pub struct Response {
    _response: *mut TRITONBACKEND_Response,
}

pub struct Output {
    _output: *mut TRITONBACKEND_Output,
}

impl Response {
    pub fn new(request: Request) -> Result<Self, ServerError> {
        let mut response: *mut TRITONBACKEND_Response =
            std::ptr::null_mut() as *mut TRITONBACKEND_Response;
        let mut error: *mut TRITONSERVER_Error = std::ptr::null_mut() as *mut TRITONSERVER_Error;
        unsafe {
            let request: *mut TRITONBACKEND_Request = request._request;
            error = TRITONBACKEND_ResponseNew(&mut response, request);
        }
        return if error.is_null() {
            Ok(Response {
                _response: response,
            })
        } else {
            Err(ServerError { _error: error })
        };
    }

    pub fn get_output(
        &self,
        name: String,
        datatype: TRITONSERVER_DataType,
        shape: Vec<i64>,
    ) -> Result<Output, ServerError> {
        let mut output: *mut TRITONBACKEND_Output =
            std::ptr::null_mut() as *mut TRITONBACKEND_Output;
        let nameRaw = CString::new(name).unwrap();
        let name: *const std::ffi::c_char = nameRaw.as_ptr();
        let mut shape = shape;
        let dims_count = shape.len() as u32;
        let err = unsafe {
         TRITONBACKEND_ResponseOutput(
                self._response,
                &mut output,
                name,
                datatype,
                shape.as_ptr(),
                dims_count,
            )
        };
        return if err.is_null() {
            Ok(Output { _output: output })
        } else {
            Err(ServerError { _error: err })
        };
    }
}
impl Output {
    pub fn get_buffer(
        &self,
        buffer_byte_size: u64,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
    ) -> Result<Buffer, ServerError> {
        let mut buffer: *mut std::ffi::c_void = std::ptr::null_mut() as *mut std::ffi::c_void;
        let mut err: *mut TRITONSERVER_Error = std::ptr::null_mut() as *mut TRITONSERVER_Error;
        let mut memory_type = memory_type;
        let mut memory_type_id = memory_type_id;
        unsafe {
            err = TRITONBACKEND_OutputBuffer(
                self._output,
                &mut buffer,
                buffer_byte_size,
                &mut memory_type as *mut u32,    //both input and output
                &mut memory_type_id as *mut i64, // both input and output
            );
        }
        return if err.is_null() {
            Ok(Buffer {
                _buffer: buffer,
                buffer_byte_size,
                memory_type,
                memory_type_id,
            })
        } else {
            Err(ServerError { _error: err })
        };
    }
}
impl Drop for Request {
    fn drop(&mut self) {
        let _err = unsafe{ TRITONBACKEND_RequestRelease(
            self._request,
            tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL,
        )};
        /* TODO: deal with error
         */
    }
}
