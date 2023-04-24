use jlrs::prelude::*;

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, Typecheck, ValidField, ConstructType, CCallArg, CCallReturn)]
#[jlrs(julia_type = "Main.InferRequest")]
pub struct InferRequest<'scope, 'data> {
    pub request_id: ::std::option::Option<::jlrs::data::managed::string::StringRef<'scope>>,
    pub correlation_id: i32,
    pub model_name: ::std::option::Option<::jlrs::data::managed::string::StringRef<'scope>>,
    pub model_version: i32,
    pub flags: i32,
    pub inputs: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    pub requested_output_names: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
}

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, Typecheck, ValidField, ConstructType, CCallArg, CCallReturn)]
#[jlrs(julia_type = "Main.InferResponse")]
pub struct InferResponse<'scope, 'data> {
    pub output_tensors: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    pub error: JbError<'scope>,
}

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, Typecheck, ValidField, ConstructType, CCallArg, CCallReturn)]
#[jlrs(julia_type = "Main.JbError")]
pub struct JbError<'scope> {
    pub message: ::std::option::Option<::jlrs::data::managed::string::StringRef<'scope>>,
}

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, Typecheck, ValidField, ConstructType, CCallArg, CCallReturn)]
#[jlrs(julia_type = "Main.JbTensor")]
pub struct JbTensor<'scope, 'data> {
    pub name: ::std::option::Option<::jlrs::data::managed::string::StringRef<'scope>>,
    pub dtype: u32,
    pub dims: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    pub memory_ptr: ::std::option::Option<::jlrs::data::managed::value::ValueRef<'scope, 'data>>,
    pub memory_type_id: i64,
    pub memory_type: u32,
    pub byte_size: u64,
}