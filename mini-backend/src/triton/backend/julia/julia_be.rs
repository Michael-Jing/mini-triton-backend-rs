#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use jlrs::convert::into_julia::IntoJulia;
    use jlrs::prelude::RuntimeBuilder;
    use jlrs::prelude::*;
    use jl_sys::*;
    use jl_sys::jl_eval_string;

   

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

                let array1 = Array::from_slice(frame.as_extended_target(), &mut data1, (2, 3) ).unwrap().unwrap().as_value();
                let array2 = Array::from_slice(frame.as_extended_target(), &mut data2, (3, 1)).unwrap().unwrap().as_value();
                

                let func2 = Module::main(&frame).function(&mut frame, "mydot" )?;
                let array_pointer = unsafe { func2.call2(&mut frame, array1, array2) }
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

