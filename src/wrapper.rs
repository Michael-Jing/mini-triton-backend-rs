use std::ffi::CString;
use std::mem::MaybeUninit;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub struct ServerError {
    _error: *mut TRITONSERVER_Error,
}
pub struct Input {
    _input: *mut TRITONBACKEND_Input,
}

pub struct InputProperties {
    name: Option<String>,
    datatype: Option<TRITONSERVER_DataType>,
    shape: Option<Vec<i64>>,
    dims_count: Option<u32>,
    byte_size: Option<u64>,
    buffer_count: Option<u32>,
}
pub struct Buffer {
    _buffer: *const std::os::raw::c_void,
    buffer_byte_size: u64,
    memory_type: TRITONSERVER_MemoryType,
    memory_type_id: i64,
}
impl Input {
    pub fn InputProperties(&self) -> Result<InputProperties, ServerError> {
        let mut name = MaybeUninit::uninit();
        let mut datatype = MaybeUninit::uninit();
        let mut shape = MaybeUninit::uninit();
        let mut dims_count = MaybeUninit::uninit();
        let mut byte_size = MaybeUninit::uninit();
        let mut buffer_count = MaybeUninit::uninit();
        let err = TRITONBACKEND_InputProperties(
            self._input,
            name.as_mut_ptr(),
            datatype.as_mut_ptr(),
            shape.as_mut_ptr(),
            dims_count.as_mut_ptr(),
            byte_size.as_mut_ptr(),
            buffer_count.as_mut_ptr(),
        );
        let dims_count = match dims_count.as_ref() {
            Some(x) => Some(x),
            None => None,
        };

        let mut shape: Option<_> = None;
        unsafe {
            match shape.as_ref() {
                Some(x) => {
                    if let Some(l) = dims_count {
                        shape = Some(std::slice::from_raw_parts(x, l));
                    }
                }
                None => None,
            }
        };

        return if err.is_null() {
            Ok(InputProperties {
                // fill in the fields
                name: match name.as_ref() {
                    Some(n) => Some(n),
                    None => None,
                },
                datatype: match datatype.as_ref() {
                    Some(t) => some(t),
                    None => None,
                },
                shape,
                dims_count,
                byte_size: match byte_size.as_ref() {
                    Some(x) => Some(x),
                    None => None,
                },
                buffer_count: match buffer_count.as_ref() {
                    Some(x) => Some(x),
                    None => None,
                },
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
        let err = TRITONBACKEND_InputBuffer(
            self._input,
            index,
            &mut buffer,
            &mut buffer_byte_size,
            &mut memory_type,
            &mut memory_type_id,
        );
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
        match TRITONBACKEND_RequestInputCount(self._request, &mut count).as_ref() {
            Some(error) => Err(ServerError { _error: error }),
            None => Ok(count),
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
        let err = TRITONBACKEND_RequestInputByIndex(self._request, index, &mut input);
        return if err.is_null() {
            Ok(Input { _input: input })
        } else {
            Err(ServerError { _error: err })
        };
    }
}

pub struct Response {
    _response: TRITONBACKEND_Response,
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
        let dims_count = shape.len();
        let mut err: *mut TRITONSERVER_Error = std::ptr::null_mut() as *mut TRITONSERVER_Error;
        unsafe {
            err = TRITONBACKEND_ResponseOutput(
                self._response,
                &mut output,
                name,
                datatype,
                &shape.as_mut_ptr(),
                dims_count,
            );
        }
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
        let _err = TRITONBACKEND_RequestRelease(
            self._request,
            tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL,
        );
        /* TODO: deal with error
         */
    }
}
