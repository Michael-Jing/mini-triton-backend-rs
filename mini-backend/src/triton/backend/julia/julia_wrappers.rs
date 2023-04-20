use jlrs::prelude::*;

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, ValidField, Typecheck)]
#[jlrs(julia_type = "Main.InferRequest")]
pub struct InferRequest<'frame, 'data> {
    pub request_id: ::std::option::Option<::jlrs::wrappers::ptr::string::StringRef<'frame>>,
    pub correlation_id: i32,
    pub model_name: ::std::option::Option<::jlrs::wrappers::ptr::string::StringRef<'frame>>,
    pub model_version: i32,
    pub flags: i32,
    pub inputs: ::std::option::Option<::jlrs::wrappers::ptr::array::ArrayRef<'frame, 'data>>,
    pub requested_output_names: ::std::option::Option<::jlrs::wrappers::ptr::array::ArrayRef<'frame, 'data>>,
}

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, ValidField, Typecheck)]
#[jlrs(julia_type = "Main.InferResponse")]
pub struct InferResponse<'frame, 'data> {
    pub output_tensors: ::std::option::Option<::jlrs::wrappers::ptr::array::ArrayRef<'frame, 'data>>,
    pub error: JbError<'frame>,
}

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, ValidField, Typecheck)]
#[jlrs(julia_type = "Main.JbError")]
pub struct JbError<'frame> {
    pub message: ::std::option::Option<::jlrs::wrappers::ptr::string::StringRef<'frame>>,
}

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, ValidField, Typecheck)]
#[jlrs(julia_type = "Main.JbTensor")]
pub struct JbTensor<'frame, 'data> {
    pub name: ::std::option::Option<::jlrs::wrappers::ptr::string::StringRef<'frame>>,
    pub dtype: u32,
    pub dims: ::std::option::Option<::jlrs::wrappers::ptr::array::ArrayRef<'frame, 'data>>,
    pub memory_ptr: ::std::option::Option<::jlrs::wrappers::ptr::value::ValueRef<'frame, 'data>>,
    pub memory_type_id: i64,
    pub memory_type: u32,
    pub byte_size: u64,
}
