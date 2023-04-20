use std::ffi::c_void;
use std::sync::Mutex;

use super::jb_exception::JuliaBackendException;
use super::jb_stub_utils::{jl_to_triton_type_v1, data_pointer, nbytes};
use jlrs::prelude::*;
use jlrs::wrappers::ptr::array::dimensions::Dims;
use triton_backend_sys::sys::*;


pub struct JbTensor {
    name_: String,
    dtype_: TRITONSERVER_DataType,
    memory_ptr_: *const c_void,
    memory_type_id_: i64,
    dims_: Vec<usize>,
    memory_type_: TRITONSERVER_MemoryType,
    byte_size_: usize,


}

impl JbTensor {
    pub fn from_jl_array(name: &str, jl_array: &Array) -> Result<Self, JuliaBackendException> {
        if name == "" {
            return Err(JuliaBackendException::new(
                "Tensor name cannot be an empty string.".to_owned(),
            ));
        }
        let dtype_ = jl_to_triton_type_v1(jl_array.element_type().cast::<DataType>().unwrap())?;
        let memory_type_ = TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
        let memory_type_id_ = 0;
        
        let byte_size_ = nbytes(jl_array);
      
        // Initialize tensor dimension
        let dims_: Vec<_> = unsafe {jl_array.dimensions().into_dimensions().as_slice().into()};
        let dims_ = dims_.into_iter().rev().collect::<Vec<_>>();
        let dims_ = dims_.into_iter().rev().collect();
        let memory_ptr_ = data_pointer(jl_array);
        return Ok(JbTensor {
            name_: name.to_owned(),
            dtype_,
            memory_ptr_,
            memory_type_id_,
            dims_,
            memory_type_,
            byte_size_
        });
    }
}