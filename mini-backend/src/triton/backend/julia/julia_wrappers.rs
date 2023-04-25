use jlrs::{prelude::*, data::managed};

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

/* 
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
} */

#[repr(C)]
#[derive(Clone, Debug, Unbox, ValidLayout, Typecheck, ValidField, ConstructType, CCallArg, CCallReturn)]
#[jlrs(julia_type = "Main.JbTensor")]
pub struct JbTensor<'scope, 'data> {
    pub name: ::std::option::Option<::jlrs::data::managed::string::StringRef<'scope>>,
    pub dtype: u32,
    pub dims: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    //    pub memory_ptr: ::std::option::Option<::jlrs::data::managed::value::ValueRef<'scope, 'data>>,
    pub memory_ptr1d: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    pub memory_ptr2d: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    pub memory_ptr3d: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    pub memory_ptr4d: ::std::option::Option<::jlrs::data::managed::array::ArrayRef<'scope, 'data>>,
    pub memory_type_id: i64,
    pub memory_type: u32,
    pub byte_size: u64,
}


#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::ffi::c_void;

    use jlrs::convert::into_julia::IntoJulia;
    use jlrs::data::managed::value::typed::AsTyped;
    use jlrs::prelude::RuntimeBuilder;
    use jlrs::prelude::*;
    use jl_sys::*;
    use jl_sys::jl_eval_string;
    use crate::triton::backend::julia::julia_wrappers::JbTensor;

    use super::InferRequest;
     use jlrs::{
        data::{
            managed::{type_var::TypeVar, union::Union, union_all::UnionAll},
            types::{
                abstract_types::{AbstractChar, AbstractString, Integer, Real},
                construct_type::{
                    ArrayTypeConstructor, ConstantIsize, ConstructType, Name, TypeVarConstructor,
                    UnionTypeConstructor,
                },
            },
        },
        prelude::*,
    };




   

    #[test]
    fn test_launch() {
        let mut julia = unsafe { RuntimeBuilder::new().start().unwrap() };
        let mut frame = StackFrame::new();
        let mut julia = julia.instance(&mut frame);
        let res = julia
            .scope(|mut frame| {
                let i = Value::new(&mut frame, 2u64);
                let j = Value::new(&mut frame, 100u32);

                
                unsafe{
                     Value::eval_string(
                    &mut frame,
                    "include(\"model.jl\")"
                    ).into_jlrs_result()?;
                }

                let mut data1 = vec![1, 2, 3, 4, 5, 6];
                let mut data2= vec![1, 2, 3];

                let mut shape1 = vec![2, 3];
                let mut shape2 = vec![3, 1];

                let array1 = Array::from_slice(frame.as_extended_target(), &mut data1, (2, 3) ).unwrap().unwrap();
                let array2 = Array::from_slice(frame.as_extended_target(), &mut data2, (3, 1)).unwrap().unwrap();
                let shape1 = Array::from_slice(frame.as_extended_target(), &mut shape1, (1, 2)).unwrap().unwrap();
                let shape2 = Array::from_slice(frame.as_extended_target(), &mut shape2, (1, 2)).unwrap().unwrap();
                let tensor1 =  JbTensor {
                    name: Some(JuliaString::new(&mut frame, "tensor1").as_ref()),
                    dtype: 0,
                    dims: Some(shape1.as_ref()),
                    memory_ptr2d: Some(array1.as_ref()), memory_type: 0, memory_type_id: 0, byte_size: 24,
                    memory_ptr1d: None,
                    memory_ptr3d: None,
                    memory_ptr4d: None,
                };
                 let tensor2 =  JbTensor {
                    name: Some(JuliaString::new(&mut frame, "tensor2").as_ref()),
                    dtype: 0,
                    dims: Some(shape2.as_ref()),
                    memory_ptr2d: Some(array2.as_ref()), memory_type: 0, memory_type_id: 0, byte_size: 12,
                    memory_ptr1d: None,
                    memory_ptr3d: None,
                    memory_ptr4d: None,

                };


                // let input_data = Array::from_slice(frame.as_extended_target(), &mut input_data, (1,2)).unwrap().unwrap();

                let julia_test_string = JuliaString::new(&mut frame, "test"); 
                let julia_test_string2 = JuliaString::new(&mut frame, "test2");
                let ty = DataType::string_type(&frame).as_value();
                let mut requested_output_names = Array::new_for(frame.as_extended_target(), (1, 2), ty).unwrap();
                unsafe { 
                   let mut mutter = requested_output_names.managed_data_mut::<StringRef>().unwrap();
                   mutter.set(0, Some(julia_test_string.as_value()));
                   mutter.set(1, Some(julia_test_string2.as_value()));
                }

                let ty = <JbTensor as ConstructType>::construct_type(frame.as_extended_target());

                let mut inputs = Array::new_for(frame.as_extended_target(), (1, 2), ty).unwrap();
                unsafe {
                    let mut mutter = inputs.managed_data_mut().unwrap();
                    mutter.set(0, Some(tensor1.as_value()));
                    mutter.set(1, Some(tensor2));

                }
               

                let infer_request = InferRequest {
                    request_id: Some(JuliaString::new(&mut frame, "0").as_ref()),
                    correlation_id: 0,
                    model_name: Some(JuliaString::new(&mut frame, "test").as_ref()),
                    model_version: 1,
                    flags: 0,
                    inputs: Some(inputs.as_ref()),
                    requested_output_names: Some(requested_output_names.as_ref()),
                };
                let func2 = Module::main(&frame).function(&mut frame, "execute" )?;
                let array_pointer = unsafe { func2.call1(&mut frame, infer_request.as_value()) }
                    .into_jlrs_result()?;
                let ans_array = array_pointer.cast::<Array>()?;
                println!("array1 is {:?}", array1);
                println!("ans array is {:?} ", ans_array);
                
                
                
                let func = Module::base(&frame).function(&mut frame, "+")?;
                unsafe { func.call2(&mut frame, i, j) }
                    .into_jlrs_result()?
                    .unbox::<u64>()
            }
        ).unwrap();
        println!("res is {:?}", res);
    }
}

