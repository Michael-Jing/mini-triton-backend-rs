use std::sync::Arc;

use jlrs::data::types::foreign_type::OpaqueType;
use jlrs::prelude::*;



use crate::triton::backend::julia::julia_wrappers::InferRequest;

use super::jb_tensor::JbTensor;
use std::collections::HashSet;
/*
#[repr(C)]
#[derive(Unbox, Typecheck, ValidLayout, Clone)]
pub struct InferRequest {
    request_id_: String,
    correlation_id_: u64,
    inputs_: Vec<Arc<JbTensor>>,
    requested_output_names_: HashSet<String>,
    model_name: String,
    model_version_: i64,
    flags_: u32, 
}


julia_module!{
    become init_infer_request;
    struct InferRequest as InferRequest;
}

*/
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

                let array1 = Array::from_slice(frame.as_extended_target(), &mut data1, (2, 3) ).unwrap().unwrap(); //.as_value();
                let array2 = Array::from_slice(frame.as_extended_target(), &mut data2, (3, 1)).unwrap().unwrap();// .as_value();
                let tensor1 = JbTensor::from_jl_array("input1", array1).unwrap();
                let tensor2 = JbTensor::from_jl_array("input2", array2).unwrap();
                let mut request_inputs = vec![];
                request_inputs.push(Arc::new(tensor1));
                request_inputs.push(Arc::new(tensor2));
                let mut requested_output_names = HashSet::new();
                requested_output_names.insert("output".to_owned());
                let infer_request = InferRequest {
                    request_id_: "0".to_owned(),
                    correlation_id_: 0,
                    inputs_: request_inputs,
                    requested_output_names_: requested_output_names,
                    model_name: "test".to_owned(),
                    model_version_: 0,
                    flags_: 0,
                };


                let func2 = Module::main(&frame).function(&mut frame, "mydot" )?;
                let array_pointer = unsafe { func2.call1(&mut frame, infer_request) }
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


