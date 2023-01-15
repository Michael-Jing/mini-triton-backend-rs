// how to deal with Request
/*
	1. get request input count
	2. use index < input_count to get input or get input by name (name could possibly be from model)
	3. Input Buffer holds data

*/
pub struct ServerError {
	_error: *mut TRITONSERVER_Error
}
pub struct Input {
	_input: *mut TRITONBACKEND_Input
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
	_buffer: *mut *const ::std::os::raw::c_void,
	 buffer_byte_size:  u64,
        memory_type:  TRITONSERVER_MemoryType,
        memory_type_id:  i64,
}
impl Input {
	pub fn InputProperties(&self) -> Result<InputProperties, E>{
		
		 let mut name = CString::new("").as_mut_ptr();
		 let mut datatype: TRITONSERVER_DataType = 0;
		 let mut shape: *const i64 = std::ptr::null() as *const i64;
		 let mut dims_count: u32 = 0;
		 let mut byte_size: u64 = 0;
		 let mut buffer_count: u32 = 0;
		 let err = TRITONBACKEND_InputProperties(self._input, name, datatype, shape, dims_count, byte_size, buffer_count);
		if err.is_null() {
			return Ok(InputProperties {
				// fill in the fields
				name: match name.as_ref() {
					Some(n) => Some(n),
					None => None ,
				},
				datatype: match datatype.as_ref() {
					Some(t) => some(t),
					None => None ,
				},
				shape: match shape.as_ref() {
					Some(x) => Some(x),
					None => None,
				},
				dims_count: match dims_count.as_ref() {
					Some(x) => Some(x),
					None => None,
				}, 
				byte_size: match byte_size.as_ref() {
					Some(x) => Some(x),
					None => None,
				}
				buffer_count : match buffer_count.as_ref() {
					Some(x) => Some(x),
					None => None,
				}
			});
		}  else {
			return Err(ServerError{_error: error});
		}
		 
	}

	
	pub fn InputBuffer(&self, index: u32) -> Result<Buffer, E> {
		/*
		 TRITONBACKEND_InputBuffer(
        input: *mut TRITONBACKEND_Input,
        index: u32,
        buffer: *mut *const ::std::os::raw::c_void,
        buffer_byte_size: *mut u64,
        memory_type: *mut TRITONSERVER_MemoryType,
        memory_type_id: *mut i64,
    ) -> *mut TRITONSERVER_Error; */
		let mut buffer: *const std::os::raw::c_void = std::ptr::null() as *const std::os::raw::c_void;
		let mut buffer_byte_size: u64 = 0;
		let mut memory_type: TRITONSERVER_MemoryType = 0;
		let mut memory_type_id: i64 = 0;
		let err = TRITONBACKEND_InputBuffer(self._input, index, &mut buffer, &mut buffer_byte_size, &mut memory_type, &mut memory_type_id);
		if err.is_null() {
			return Ok(Buffer{
				_buffer: buffer,
				buffer_byte_size: buffer_byte_size,
				memory_type: memory_type,
				memory_type_id: memory_type_id,
			});
		} else {
			return Err(ServerError(_error: err));
		}
	}
}

pub struct Request {
	_request:  *mut TRITONBACKEND_Request
}
impl Request {
	pub fn InputCount(&self) -> Result<u32, E> {
		let mut count = 0;
		match TRITONBACKEND_RequestInputCount (
       self._request,
       &mut count 
    ).as_ref() {
		Some(error) => Err(ServerError{_error: error}),
		None => Ok(count),
	}
	}

	pub fn InputByIndex(&self, index: u32) -> Result<Input, E>{
		/*
		 TRITONBACKEND_RequestInputByIndex(
        request: *mut TRITONBACKEND_Request,
        index: u32,
        input: *mut *mut TRITONBACKEND_Input,
    ) -> *mut TRITONSERVER_Error;
	 */
	 let mut input: *mut TRITONBACKEND_Input = std::ptr::null_mut() as *mut TRITONBACKEND_Input;
	 let err = TRITONBACKEND_RequestInputByIndex(self._request, &mut input);
	 if err.is_null() {
		return Ok(Input{_input: input});
	 } else {
		return Err(ServerError{_error: error});
	 }
	}
	
	
}

impl Drop for Request {
    fn drop(&mut self) {
		let err = TRITONBACKEND_RequestRelease(self._request, tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL);
		/* TODO: deal with error
		 */
    }
}
