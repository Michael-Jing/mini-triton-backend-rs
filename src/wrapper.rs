use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem;
use std::mem::MaybeUninit;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[derive(Debug)]
pub struct TritonError {
    _err: *mut TRITONSERVER_Error,
}
#[derive(Debug)]
pub struct Error {
    msg: String,
}
pub struct Input {
    _input: *mut TRITONBACKEND_Input,
}

#[derive(Debug)]
pub struct InputProperties<'a> {
    name: &'a str,
    datatype: TRITONSERVER_DataType,
    shape: Option<&'a [i64]>,
    dims_count: u32,
    byte_size: u64,
    buffer_count: u32,
    phantom: PhantomData<&'a Input>,
}

pub struct ResponseFactory {
    _factory: *mut TRITONBACKEND_ResponseFactory,
}

pub struct OutputBuffer {
    pub _buffer: *const std::os::raw::c_void,
    buffer_byte_size: u64,
    memory_type: TRITONSERVER_MemoryType,
    memory_type_id: i64,
    datatype: TRITONSERVER_DataType,
}

pub struct InputBuffer<'a, T> {
    pub _buffer: *const T,
    offset: isize,
    pub buffer_byte_size: u64,
    memory_type: TRITONSERVER_MemoryType,
    memory_type_id: i64,
    phantom: PhantomData<&'a Input>,
}

impl<'a, T> Iterator for InputBuffer<'a, T>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if ((self.offset + 1) as usize * mem::size_of::<T>()) as u64 <= self.buffer_byte_size {
            let current = unsafe { (*self._buffer.offset(self.offset)).clone() };
            self.offset += 1;
            Some(current)
        } else {
            None
        }
    }
}

impl OutputBuffer {
    pub fn write<T>(&self, data: Vec<T>) -> Result<(), Error> {
        if data.is_empty() {
            return Err(Error {
                msg: "data is empty".to_owned(),
            });
        }
        let ele_size = std::mem::size_of_val(&data[0]);
        if ele_size * data.len() > self.buffer_byte_size as usize {
            return Err(Error {
                msg: "buffer size is too small to hold all data".to_owned(),
            });
        }
        let mut buffer = self._buffer as *mut T;
        unsafe {
            let begin = buffer;
            for (i, d) in data.into_iter().enumerate() {
                *begin.offset(i as isize) = d;
            }
        }
        Ok(())
    }

    pub fn write_fp32(&self, data: Vec<f32>) -> Result<(), Error> {
        if data.is_empty() {
            return Err(Error {
                msg: "data is empty".to_owned(),
            });
        }
        let ele_size = std::mem::size_of_val(&data[0]);
        if ele_size * data.len() > self.buffer_byte_size as usize {
            return Err(Error {
                msg: "buffer size is too small to hold all data".to_owned(),
            });
        }
        let mut buffer = self._buffer as *mut f32;
        unsafe {
            let begin = buffer;
            for (i, d) in data.into_iter().enumerate() {
                *begin.offset(i as isize) = d;
            }
        }
        Ok(())
    }
}
impl Input {
    pub fn get_input_properties_for_host_policy() {
        todo!()
    }
    pub fn get_input_properties(&self) -> Result<InputProperties, TritonError> {
        let mut name: *mut *const std::os::raw::c_char =
            Box::into_raw(Box::new(0)) as *mut *const std::os::raw::c_char;
        let mut datatype: TRITONSERVER_DataType = 0;
        let mut shapeRaw: *mut *const i64 = Box::into_raw(Box::new(0)) as *mut *const i64;
        let mut dims_count: u32 = 0;
        let mut byte_size: u64 = 0;
        let mut buffer_count: u32 = 0;

        let err = unsafe {
            TRITONBACKEND_InputProperties(
                self._input,
                name,
                &mut datatype,
                shapeRaw,
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            )
        };

        let shape = if shapeRaw.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(*shapeRaw, dims_count as usize) })
        };

        return if err.is_null() {
            Ok(InputProperties {
                // fill in the fields
                name: unsafe { CStr::from_ptr(*name).to_str().unwrap_or("") },

                datatype,
                shape,
                dims_count,
                byte_size,
                buffer_count,
                phantom: PhantomData,
            })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_buffer<T>(
        &self,
        index: u32,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
        example_value: T,
    ) -> Result<InputBuffer<T>, TritonError> {
        let mut buffer: *const std::os::raw::c_void =
            std::ptr::null() as *const std::os::raw::c_void;
        let mut buffer_byte_size: u64 = 0;
        let mut memory_type: TRITONSERVER_MemoryType = memory_type;
        let mut memory_type_id: i64 = memory_type_id;
        let err = unsafe {
            TRITONBACKEND_InputBuffer(
                self._input,
                index,
                &mut buffer,
                &mut buffer_byte_size,
                &mut memory_type,
                &mut memory_type_id,
            )
        };
        /* let buffer = match datatype {
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => buffer as *const bool,
          TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => buffer as *const u8,
        }; */
        return if err.is_null() {
            Ok(InputBuffer {
                _buffer: buffer as *const T,
                offset: 0,
                buffer_byte_size,
                memory_type,
                memory_type_id,
                phantom: PhantomData,
            })
        } else {
            Err(TritonError { _err: err })
        };
    }
}

pub struct Request {
    _request: *mut TRITONBACKEND_Request,
}
impl Request {
    pub fn new(request: *mut TRITONBACKEND_Request) -> Result<Self, TritonError> {
        return Ok(Request { _request: request });
    }
    pub fn get_input_count(&self) -> Result<u32, TritonError> {
        let mut count = 0;
        let err = unsafe { TRITONBACKEND_RequestInputCount(self._request, &mut count) };
        return if err.is_null() {
            Ok(count)
        } else {
            Err(TritonError { _err: err })
        };
    }
    pub fn get_output_count(&self) -> Result<u32, TritonError> {
        let mut count = 0;
        let err = unsafe { TRITONBACKEND_RequestOutputCount(self._request, &mut count) };
        return if err.is_null() {
            Ok(count)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_input_by_index(&self, index: u32) -> Result<Input, TritonError> {
        /*
             TRITONBACKEND_RequestInputByIndex(
            request: *mut TRITONBACKEND_Request,
            index: u32,
            input: *mut *mut TRITONBACKEND_Input,
        ) -> *mut TRITONSERVER_err;
         */
        let mut input: *mut TRITONBACKEND_Input = std::ptr::null_mut() as *mut TRITONBACKEND_Input;
        let err = unsafe { TRITONBACKEND_RequestInputByIndex(self._request, index, &mut input) };
        return if err.is_null() {
            Ok(Input { _input: input })
        } else {
            Err(TritonError { _err: err })
        };
    }
    pub fn get_input(&self, name: &str) -> Result<Input, TritonError> {
        /* TRITONBACKEND_RequestInput(
        struct TRITONBACKEND_Request* request, const char* name,
        struct TRITONBACKEND_Input** input); */
        let mut input: *mut TRITONBACKEND_Input = std::ptr::null_mut() as *mut TRITONBACKEND_Input;
        let name = CString::new(name).unwrap();
        let err = unsafe { TRITONBACKEND_RequestInput(self._request, name.as_ptr(), &mut input) };
        return if err.is_null() {
            Ok(Input { _input: input })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_input_name(&self, index: u32) -> Result<&str, TritonError> {
        /* TRITONBACKEND_RequestInputName(
        struct TRITONBACKEND_Request* request, const uint32_t index,
        const char** input_name); */
        let name = CString::new("").unwrap();
        let mut name = name.as_ptr();
        let err = unsafe { TRITONBACKEND_RequestInputName(self._request, index, &mut name) };
        return if err.is_null() {
            let name = unsafe { CStr::from_ptr(name).to_str().unwrap_or("") };
            Ok(name)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_output_name(&self, index: u32) -> Result<&str, TritonError> {
        let name = CString::new("").unwrap();
        let mut name = name.as_ptr();
        let err = unsafe { TRITONBACKEND_RequestOutputName(self._request, index, &mut name) };
        return if err.is_null() {
            let name = unsafe { CStr::from_ptr(name).to_str().unwrap_or("") };
            Ok(name)
        } else {
            Err(TritonError { _err: err })
        };
    }
    pub fn get_id(&self) -> Result<&str, TritonError> {
        let id = CString::new("").unwrap();
        let mut id = id.as_ptr();
        let err = unsafe { TRITONBACKEND_RequestId(self._request, &mut id) };
        return if err.is_null() {
            let id = unsafe { CStr::from_ptr(id).to_str().unwrap_or("") };
            Ok(id)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_correlation_id(&self) -> Result<u64, TritonError> {
        let mut id = 0;
        let err = unsafe { TRITONBACKEND_RequestCorrelationId(self._request, &mut id) };
        return if err.is_null() {
            Ok(id)
        } else {
            Err(TritonError { _err: err })
        };
    }
    pub fn get_correlation_id_string(&self) -> Result<&str, TritonError> {
        let id = CString::new("").unwrap();
        let mut id = id.as_ptr();
        let err = unsafe { TRITONBACKEND_RequestCorrelationIdString(self._request, &mut id) };
        return if err.is_null() {
            let id = unsafe { CStr::from_ptr(id).to_str().unwrap_or("") };
            Ok(id)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_flags(&self) -> Result<u32, TritonError> {
        let mut id = 0;
        let err = unsafe { TRITONBACKEND_RequestFlags(self._request, &mut id) };
        return if err.is_null() {
            Ok(id)
        } else {
            Err(TritonError { _err: err })
        };
    }
    pub fn get_output_buffer_properties(
        &self,
        name: Option<&str>,
        byte_size: Option<&mut usize>,
        memory_type: &mut TRITONSERVER_MemoryType,
        memory_type_id: &mut i64,
    ) -> Result<(), TritonError> {
        let name = match name {
            Some(str) => CString::new(str).unwrap().as_ptr(),
            None => std::ptr::null() as *const i8,
        };
        let byte_size = match byte_size {
            Some(size) => size,
            None => std::ptr::null_mut() as *mut usize,
        };
        let err = unsafe {
            TRITONBACKEND_RequestOutputBufferProperties(
                self._request,
                name,
                byte_size,
                memory_type,
                memory_type_id,
            )
        };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }
}

pub struct Response {
    pub _response: *mut TRITONBACKEND_Response,
}

pub struct Output {
    _output: *mut TRITONBACKEND_Output,
}

impl ResponseFactory {
    pub fn new(request: Request) -> Result<Self, TritonError> {
        let mut factory: *mut TRITONBACKEND_ResponseFactory =
            std::ptr::null_mut() as *mut TRITONBACKEND_ResponseFactory;
        let err = unsafe { TRITONBACKEND_ResponseFactoryNew(&mut factory, request._request) };
        return if err.is_null() {
            Ok(ResponseFactory { _factory: factory })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn send_flags(&self, flags: u32) -> Result<(), TritonError> {
        let err = unsafe { TRITONBACKEND_ResponseFactorySendFlags(self._factory, flags) };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }
}

impl Drop for ResponseFactory {
    fn drop(&mut self) {
        let err = unsafe { TRITONBACKEND_ResponseFactoryDelete(self._factory) };
        if !err.is_null() {
            //TODO: add log maybe
        }
    }
}
impl Response {
    pub fn new(request: &Request) -> Result<Self, TritonError> {
        let mut response: *mut TRITONBACKEND_Response =
            std::ptr::null_mut() as *mut TRITONBACKEND_Response;
        let err = unsafe { TRITONBACKEND_ResponseNew(&mut response, request._request) };
        return if err.is_null() {
            Ok(Response {
                _response: response,
            })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn from_factor(factory: &ResponseFactory) -> Result<Self, TritonError> {
        let mut response = std::ptr::null_mut() as *mut TRITONBACKEND_Response;
        let err = unsafe { TRITONBACKEND_ResponseNewFromFactory(&mut response, factory._factory) };
        return if err.is_null() {
            Ok(Response {
                _response: response,
            })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_output(
        &self,
        name: String,
        datatype: TRITONSERVER_DataType,
        shape: Vec<i64>,
    ) -> Result<Output, TritonError> {
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
            Err(TritonError { _err: err })
        };
    }
    pub fn send(
        &self,
        complete_flag: TRITONSERVER_ResponseCompleteFlag,
        err: Option<TritonError>,
    ) -> Result<(), TritonError> {
        let err = unsafe {
            // notes for safe wrap: TRITONBACKEND_ResponseSend  takes ownership of response;
            TRITONBACKEND_ResponseSend(
                self._response,
                complete_flag,
                match err {
                    Some(e) => e._err,
                    None => std::ptr::null_mut() as *mut TRITONSERVER_Error,
                },
            )
        };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_string_parameter(&self, name: &str, value: &str) -> Result<(), TritonError> {
        let name = CString::new(name).unwrap();
        let value = CString::new(value).unwrap();
        let err = unsafe {
            TRITONBACKEND_ResponseSetStringParameter(self._response, name.as_ptr(), value.as_ptr())
        };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_int_parameter(&self, name: &str, value: i64) -> Result<(), TritonError> {
        let name = CString::new(name).unwrap();
        let err =
            unsafe { TRITONBACKEND_ResponseSetIntParameter(self._response, name.as_ptr(), value) };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_bool_parameter(&self, name: &str, value: bool) -> Result<(), TritonError> {
        let name = CString::new(name).unwrap();
        let err =
            unsafe { TRITONBACKEND_ResponseSetBoolParameter(self._response, name.as_ptr(), value) };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }
}
impl Output {
    pub fn get_buffer(
        &self,
        buffer_byte_size: u64,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
        datatype: TRITONSERVER_DataType,
    ) -> Result<OutputBuffer, TritonError> {
        let mut buffer: *mut std::ffi::c_void = std::ptr::null_mut() as *mut std::ffi::c_void;
        let mut memory_type = memory_type;
        let mut memory_type_id = memory_type_id;
        let err = unsafe {
            TRITONBACKEND_OutputBuffer(
                self._output,
                &mut buffer,
                buffer_byte_size,
                &mut memory_type as *mut u32,    //both input and output
                &mut memory_type_id as *mut i64, // both input and output
            )
        };
        return if err.is_null() {
            Ok(OutputBuffer {
                _buffer: buffer,
                buffer_byte_size,
                memory_type,
                memory_type_id,
                datatype,
            })
        } else {
            Err(TritonError { _err: err })
        };
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        let err = unsafe { TRITONBACKEND_ResponseDelete(self._response) };
        if !err.is_null() {
            //TODO: add logging
        }
    }
}
impl Drop for Request {
    fn drop(&mut self) {
        let _err = unsafe {
            /* according to tritonbackend.h, there might be a benefit to release a re
            quest as early as possible to release all it's resources */
            TRITONBACKEND_RequestRelease(
                self._request,
                tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL,
            )
        };
        /* TODO: deal with error
         */
    }
}

/*

TRITONBACKEND_DECLSPEC struct TRITONSERVER_err* TRITONBACKEND_ApiVersion(
    uint32_t* major, uint32_t* minor); */
pub struct TritonApiVersion {
    major: u32,
    minor: u32,
}
pub fn get_backend_api_version() -> Result<TritonApiVersion, TritonError> {
    let mut major: u32 = 0;
    let mut minor: u32 = 0;
    let err = unsafe { TRITONBACKEND_ApiVersion(&mut major, &mut minor) };
    return if err.is_null() {
        Ok(TritonApiVersion { major, minor })
    } else {
        Err(TritonError { _err: err })
    };
}

pub struct InstanceState {}
pub struct ModelInstance {
    _instance: *mut TRITONBACKEND_ModelInstance,
}

impl ModelInstance {
    pub fn report_statistics(
        &self,
        request: &Request,
        success: bool,
        exec_start_ns: u64,
        compute_start_ns: u64,
        compute_end_ns: u64,
        exec_end_ns: u64,
    ) -> Result<(), TritonError> {
        let err = unsafe {
            TRITONBACKEND_ModelInstanceReportStatistics(
                self._instance,
                request._request,
                success,
                exec_start_ns,
                compute_start_ns,
                compute_end_ns,
                exec_end_ns,
            )
        };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }
    pub fn report_batch_statistics(
        &self,
        batch_size: u64,
        exec_start_ns: u64,
        compute_start_ns: u64,
        compute_end_ns: u64,
        exec_end_ns: u64,
    ) -> Result<(), TritonError> {
        let err = unsafe {
            TRITONBACKEND_ModelInstanceReportBatchStatistics(
                self._instance,
                batch_size,
                exec_start_ns,
                compute_start_ns,
                compute_end_ns,
                exec_end_ns,
            )
        };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }
    pub fn get_state(&self) -> Result<&InstanceState, TritonError> {
        let mut state = std::ptr::null_mut() as *mut std::os::raw::c_void;
        let err = unsafe { TRITONBACKEND_ModelInstanceState(self._instance, &mut state) };
        return if err.is_null() {
            let state = state as *mut InstanceState;
            Ok(unsafe { &*state })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_state(&self, state: &mut InstanceState) -> Result<(), TritonError> {
        let state = state as *mut InstanceState;
        let err = unsafe {
            TRITONBACKEND_ModelInstanceSetState(self._instance, state as *mut std::os::raw::c_void)
        };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_model(&self) -> Result<Model, TritonError> {
        let mut model = std::ptr::null_mut() as *mut TRITONBACKEND_Model;
        let err = unsafe { TRITONBACKEND_ModelInstanceModel(self._instance, &mut model) };
        return if err.is_null() {
            Ok(Model { _model: model })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_secondary_device_properties(
        &self,
        index: u32,
    ) -> Result<DeviceProperty, TritonError> {
        let mut kind = CString::new("").unwrap();
        let mut kind = kind.as_ptr();
        let mut id = 0;
        let err = unsafe {
            TRITONBACKEND_ModelInstanceSecondaryDeviceProperties(
                self._instance,
                index,
                &mut kind,
                &mut id,
            )
        };
        return if err.is_null() {
            Ok(DeviceProperty {
                kind: unsafe { CStr::from_ptr(kind).to_str().unwrap_or("") },
                id,
            })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_secondary_device_count(&self) -> Result<u32, TritonError> {
        let mut count = 0;
        let err =
            unsafe { TRITONBACKEND_ModelInstanceSecondaryDeviceCount(self._instance, &mut count) };
        return if err.is_null() {
            Ok(count)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_profile_count(&self) -> Result<u32, TritonError> {
        let mut count = 0;
        let err = unsafe { TRITONBACKEND_ModelInstanceProfileCount(self._instance, &mut count) };
        return if err.is_null() {
            Ok(count)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_profile_name(&self, index: u32) -> Result<&str, TritonError> {
        let mut name = CString::new("").unwrap();
        let mut name = name.as_ptr();
        let err =
            unsafe { TRITONBACKEND_ModelInstanceProfileName(self._instance, index, &mut name) };
        return if err.is_null() {
            Ok(unsafe { CStr::from_ptr(name).to_str().unwrap_or("") })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn is_passive(&self) -> Result<bool, TritonError> {
        let mut passive = false;
        let err = unsafe { TRITONBACKEND_ModelInstanceIsPassive(self._instance, &mut passive) };
        return if err.is_null() {
            Ok(passive)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_host_policy(&self) -> Result<TritonMessage, TritonError> {
        let mut policy = std::ptr::null_mut() as *mut TRITONSERVER_Message;
        let err = unsafe { TRITONBACKEND_ModelInstanceHostPolicy(self._instance, &mut policy) };
        return if err.is_null() {
            Ok(TritonMessage { _message: policy })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_device_id(&self) -> Result<i32, TritonError> {
        let mut id = 0;
        let err = unsafe { TRITONBACKEND_ModelInstanceDeviceId(self._instance, &mut id) };
        return if err.is_null() {
            Ok(id)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_kind(&self) -> Result<TRITONSERVER_InstanceGroupKind, TritonError> {
        let mut kind = TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_AUTO;
        let err = unsafe { TRITONBACKEND_ModelInstanceKind(self._instance, &mut kind) };
        return if err.is_null() {
            Ok(kind)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_name(&self) -> Result<&str, TritonError> {
        let mut name = CString::new("").unwrap();
        let mut name = name.as_ptr();
        let err = unsafe { TRITONBACKEND_ModelInstanceName(self._instance, &mut name) };
        return if err.is_null() {
            Ok(unsafe { CStr::from_ptr(name).to_str().unwrap_or("") })
        } else {
            Err(TritonError { _err: err })
        };
    }
}

pub struct Model {
    _model: *mut TRITONBACKEND_Model,
}

pub struct DeviceProperty<'a> {
    kind: &'a str,
    id: i64,
}

pub struct TritonMessage {
    _message: *mut TRITONSERVER_Message,
}
pub struct Backend {
    _backend: *mut TRITONBACKEND_Backend,
}

pub struct Server {
    _server: *mut TRITONSERVER_Server,
}

impl Backend {
    pub fn get_name(&self) -> Result<&str, TritonError> {
        let mut name = CString::new("").unwrap();
        let mut name = name.as_ptr();
        let err = unsafe { TRITONBACKEND_BackendName(self._backend, &mut name) };
        return if err.is_null() {
            Ok(unsafe { CStr::from_ptr(name).to_str().unwrap_or("") })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_config(&self) -> Result<TritonMessage, TritonError> {
        let mut config = std::ptr::null_mut() as *mut TRITONSERVER_Message;
        let err = unsafe { TRITONBACKEND_BackendConfig(self._backend, &mut config) };
        return if err.is_null() {
            Ok(TritonMessage { _message: config })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_execution_policy(&self) -> Result<TRITONBACKEND_ExecutionPolicy, TritonError> {
        let mut policy = TRITONBACKEND_execpolicy_enum_TRITONBACKEND_EXECUTION_BLOCKING;
        let err = unsafe { TRITONBACKEND_BackendExecutionPolicy(self._backend, &mut policy) };
        return if err.is_null() {
            Ok(policy)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_execution_policy(
        &self,
        policy: TRITONBACKEND_ExecutionPolicy,
    ) -> Result<(), TritonError> {
        let err = unsafe { TRITONBACKEND_BackendSetExecutionPolicy(self._backend, policy) };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_artifacts(&self) -> Result<(TRITONBACKEND_ArtifactType, &str), TritonError> {
        let mut artifact_type = TRITONBACKEND_artifacttype_enum_TRITONBACKEND_ARTIFACT_FILESYSTEM;
        let mut location = CString::new("").unwrap();
        let mut location = location.as_ptr();
        let err = unsafe {
            TRITONBACKEND_BackendArtifacts(self._backend, &mut artifact_type, &mut location)
        };
        return if err.is_null() {
            Ok((artifact_type, unsafe {
                CStr::from_ptr(location).to_str().unwrap_or("")
            }))
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_state(&self) -> Result<&BackendState, TritonError> {
        let mut state = std::ptr::null_mut() as *mut std::os::raw::c_void;
        let err = unsafe { TRITONBACKEND_BackendState(self._backend, &mut state) };
        return if err.is_null() {
            let state = state as *mut BackendState;
            Ok(unsafe { &*state })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_state(&self, state: &BackendState) -> Result<(), TritonError> {
        let state = state as *const BackendState;
        let err =
            unsafe { TRITONBACKEND_BackendSetState(self._backend, state as *mut ffi::c_void) };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }
}

pub struct MemoryManager {
    _manager: *mut TRITONBACKEND_MemoryManager,
}

impl MemoryManager {
    pub fn new(backend: &Backend) -> Result<Self, TritonError> {
        let mut manager = std::ptr::null_mut() as *mut TRITONBACKEND_MemoryManager;
        let err = unsafe { TRITONBACKEND_BackendMemoryManager(backend._backend, &mut manager) };
        return if err.is_null() {
            Ok(MemoryManager { _manager: manager })
        } else {
            Err(TritonError { _err: err })
        };
    }
}

pub struct BackendState {}
pub struct ModelState {}

impl Model {
    pub fn get_name(&self) -> Result<&str, TritonError> {
        let name = CString::new("").unwrap();
        let mut name = name.as_ptr();
        let err = unsafe { TRITONBACKEND_ModelName(self._model, &mut name) };
        return if err.is_null() {
            Ok(unsafe { CStr::from_ptr(name).to_str().unwrap_or("") })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_version(&self) -> Result<u64, TritonError> {
        let mut version = 0;
        let err = unsafe { TRITONBACKEND_ModelVersion(self._model, &mut version) };
        return if err.is_null() {
            Ok(version)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_artifacts(&self) -> Result<(TRITONBACKEND_ArtifactType, &str), TritonError> {
        let mut artifact_type = TRITONBACKEND_artifacttype_enum_TRITONBACKEND_ARTIFACT_FILESYSTEM;
        let mut location = CString::new("").unwrap();
        let mut location = location.as_ptr();
        let err = unsafe {
            TRITONBACKEND_ModelRepository(self._model, &mut artifact_type, &mut location)
        };
        return if err.is_null() {
            Ok((artifact_type, unsafe {
                CStr::from_ptr(location).to_str().unwrap_or("")
            }))
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_config(&self, config_version: u32) -> Result<TritonMessage, TritonError> {
        let mut config = std::ptr::null_mut() as *mut TRITONSERVER_Message;
        let err = unsafe { TRITONBACKEND_ModelConfig(self._model, config_version, &mut config) };
        return if err.is_null() {
            Ok(TritonMessage { _message: config })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn should_auto_complete_config(&self) -> Result<bool, TritonError> {
        let mut auto_complete = false;
        let err = unsafe { TRITONBACKEND_ModelAutoCompleteConfig(self._model, &mut auto_complete) };
        return if err.is_null() {
            Ok(auto_complete)
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_config(
        &self,
        config_version: u32,
        config: TritonMessage,
    ) -> Result<(), TritonError> {
        let err =
            unsafe { TRITONBACKEND_ModelSetConfig(self._model, config_version, config._message) };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_server(&self) -> Result<Server, TritonError> {
        let mut server = std::ptr::null_mut() as *mut TRITONSERVER_Server;
        let err = unsafe { TRITONBACKEND_ModelServer(self._model, &mut server) };
        return if err.is_null() {
            Ok(Server { _server: server })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_backend(&self) -> Result<Backend, TritonError> {
        let mut backend = std::ptr::null_mut() as *mut TRITONBACKEND_Backend;
        let err = unsafe { TRITONBACKEND_ModelBackend(self._model, &mut backend) };
        return if err.is_null() {
            Ok(Backend { _backend: backend })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn get_state(&self) -> Result<&ModelState, TritonError> {
        let mut state = std::ptr::null_mut() as *mut std::os::raw::c_void;
        let err = unsafe { TRITONBACKEND_ModelState(self._model, &mut state) };
        return if err.is_null() {
            let state = state as *mut ModelState;
            Ok(unsafe { &*state })
        } else {
            Err(TritonError { _err: err })
        };
    }

    pub fn set_state(&self, state: &mut ModelState) -> Result<(), TritonError> {
        let state = state as *mut ModelState;
        let err =
            unsafe { TRITONBACKEND_ModelSetState(self._model, state as *mut std::os::raw::c_void) };
        return if err.is_null() {
            Ok(())
        } else {
            Err(TritonError { _err: err })
        };
    }
}
