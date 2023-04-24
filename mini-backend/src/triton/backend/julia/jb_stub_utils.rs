use std::{ffi::{c_void, self}, slice};

use jlrs::{prelude::*, data::managed::array::dimensions::Dims};
use triton_backend_sys::sys::*;
use ndarray;

use super::jb_exception::JuliaBackendException; // {TRITONSERVER_DataType, TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64, TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL};

pub fn jl_to_triton_type_v1(
    data_type: DataType,
) -> Result<TRITONSERVER_datatype_enum, JuliaBackendException> {
    if data_type.is::<bool>() {
         return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL);
        
    }
    if data_type.is::<u8>() {
        return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8);
    }
    if data_type.is::<u16>() {
      return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16);
    }
    if data_type.is::<u32>() {
         return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32);
    }

    if data_type.is::<u64>() {
         return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64);
    }
    if data_type.is::<i8>() {
     return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8);
    }
    if data_type.is::<i16>() {
     return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16);
    }
    if data_type.is::<i32>() {
     return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32);
    }
    if data_type.is::<i64>() {
     return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64);
    }
    if data_type.is::<half::f16>() { 
         return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16);
    } 
    if data_type.is::<f32>() {
         return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32);
    }
    if data_type.is::<f64>() {
        return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64);
    }
    if data_type.is::<char>() { // TODO: found correct corresponding types for _object and bytes in numpy
         return Ok(TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES);
    }
    else {
            return Err(JuliaBackendException::new(
                "NumPy dtype is not supported.".to_owned(),
            ));
        }
    }
 
pub fn nbytes(array: &Array) -> usize {
    let num_elements = unsafe {array.dimensions().into_dimensions().size()};
    let element_size = array.element_size();
    return num_elements * element_size;
}

pub fn data_pointer(array: &Array) -> *const ffi::c_void {
    let datatype = array.element_type().cast::<DataType>().unwrap();
    unsafe {


       
    let data =
     if datatype.is::<bool>() {
        array.bits_data::<bool>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<u8>() {
        array.bits_data::<u8>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<u16>() {
        array.bits_data::<u16>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<u32>() {
        array.bits_data::<u32>().unwrap().as_slice().as_ptr() as *const c_void 
     } else if datatype.is::<u64>() {
        array.bits_data::<u64>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<i8>() {
        array.bits_data::<i8>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<u16>() {
        array.bits_data::<u16>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<u32>() {
        array.bits_data::<u32>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<u64>() {
        array.bits_data::<u64>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<half::f16>() {
        array.bits_data::<half::f16>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<f32>() {
        array.bits_data::<f32>().unwrap().as_slice().as_ptr() as *const c_void
     } else if datatype.is::<f64>() {
        array.bits_data::<f64>().unwrap().as_slice().as_ptr() as *const c_void
     } else {
        array.bits_data::<char>().unwrap().as_slice().as_ptr() as *const c_void
     }; 
     return data;
     // TODO: deal with other types
     // TODO: the data is column major, so everytime julia gets data, julia need to transpose

    }
}

/* 
pub fn offset2index(offset: usize, stride: &Vec<usize>) -> Vec<usize> {
    let N = stride.len()
    let mut index = vec![0; stride.len()];
    if stride[0] < stride[N - 1] {

    }

}
pub fn major_row2col<T>(buffer: &mut Vec<T>, shape: &Vec<usize>) {
    let N = shape.len();
    let mut cur_stride = vec![1;shape.len()];
    for i in (0..(shape.len() - 1)).rev() {
        cur_stride[i] = cur_stride[i + 1] * shape[i + 1];
    }
    let mut target_stride = vec![1; N];
    for i in 1..N {
        target_stride[i] = target_stride[i - 1] * shape[i - 1];
    }
    let size:usize = shape.iter().product();
    let mut done = vec![false; size];
    let mut temp = None;
    for i in 0..size {
        if !done[i] {
            temp = Some(buffer[i].clone());


        }

    }



}
*/
#[cfg(test)]
mod tests {
    use jl_sys::jl_init;
    use jl_sys::*;
    use jlrs::convert::ndarray::NdArrayView;
    use jlrs::data::managed::array::dimensions::Dims;
    use jlrs::memory::target::frame::GcFrame;
    use jlrs::prelude::*;
    use jlrs::prelude::{Array, Julia, RuntimeBuilder, TypedArray, Value};

    use super::jl_to_triton_type_v1;
    use triton_backend_sys::sys::*;

     #[test]
    fn test_dims() {
         let mut julia = unsafe { RuntimeBuilder::new().start().unwrap() };
        let mut frame = StackFrame::new();
        let mut julia = julia.instance(&mut frame);
        let res = julia
            .scope(|mut frame| {
                // Create the two arguments.
                let i = Value::new(&mut frame, 2u64);
                let j = Value::new(&mut frame, 1u32);

                let vec: Vec<i32> = vec![1, 2, 3];
                // The `+` function can be found in the base module.
                let func = Module::base(&frame).function(&mut frame, "+")?;
                let array = TypedArray::from_vec(frame.as_extended_target(), vec,  3).unwrap().unwrap();
                
                let dimensions = unsafe{ array.dimensions().into_dimensions() };
                let sz = dimensions.size();
                let dims = dimensions.as_slice();
                println!("dims is {:?}", dims);
                println!("size is {:?}", sz);

                
                    

                unsafe { func.call2(&mut frame, i, j) }
                    .into_jlrs_result()?
                    .unbox::<u64>()
            })
            .unwrap();
        
    }
    
    #[test]
    fn playground() {
        unsafe {
            jl_init();

            let array_type = jl_apply_array_type(jl_float64_type as *mut jl_value_t, 1);
            let array = jl_alloc_array_1d(array_type, 10);

            let elem_type = jl_typeof(array as *mut jl_value_t);

            println!("element type is {:?}", *elem_type);

        }
    }

    #[test]
    fn test_size() {
         let mut julia = unsafe { RuntimeBuilder::new().start().unwrap() };
        let mut frame = StackFrame::new();
        let mut julia = julia.instance(&mut frame);
        let res = julia
            .scope(|mut frame| {
                // Create the two arguments.
                let i = Value::new(&mut frame, 2u64);
                let j = Value::new(&mut frame, 1u32);

                let vec: Vec<i32> = vec![1, 2, 3];
                // The `+` function can be found in the base module.
                let func = Module::base(&frame).function(&mut frame, "+")?;
                let array = TypedArray::from_vec(frame.as_extended_target(), vec, 3).unwrap().unwrap();
                
                let dimensions = unsafe{ array.dimensions().into_dimensions() };
                let sz = dimensions.size();
                println!("size is {:?}", sz);

                
                    

                unsafe { func.call2(&mut frame, i, j) }
                    .into_jlrs_result()?
                    .unbox::<u64>()
            })
            .unwrap();
        
    }

    #[test]
    fn test_type() {
        let mut julia = unsafe { RuntimeBuilder::new().start().unwrap() };
        let mut frame = StackFrame::new();
        let mut julia = julia.instance(&mut frame);
        let res = julia
            .scope(|mut frame| {
                // Create the two arguments.
                let i = Value::new(&mut frame, 2u64);
                let j = Value::new(&mut frame, 1u32);

                let vec: Vec<i32> = vec![1, 2, 3];
                // The `+` function can be found in the base module.
                let func = Module::base(&frame).function(&mut frame, "+")?;
                let array = TypedArray::from_vec(frame.as_extended_target(), vec, 3).unwrap().unwrap();

                // Call the function and unbox the result as a `u64`. The result of the function
                // call is a nested `Result`; the outer error doesn't contain to any Julia
                // data, while the inner error contains the exception if one is thrown. Here the
                // exception is converted to the outer error type by calling `into_jlrs_result`, this new
                // error contains the error message Julia would have shown.

                let datatype = array.element_type().cast::<DataType>().unwrap();
                // let ptr = array.bits_data();
                // println!("datatype name is {:?}", datatype);
                let triton_type = jl_to_triton_type_v1(datatype);
                println!("data type is {:?}", triton_type);
                match triton_type {
                    Ok(dtype) => assert_eq!(dtype, TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 ),
                    Err(_) => panic!("error"),
                }

                unsafe { func.call2(&mut frame, i, j) }
                    .into_jlrs_result()?
                    .unbox::<u64>()
            })
            .unwrap();
    }
    use ndarray::ShapeBuilder;
    #[test]
    fn test_ndarray() {
        let vec = (0..12).collect::<Vec<_>>();
        let ptr = vec.as_ptr();
        let array = unsafe{ ndarray::ArrayView::from_shape_ptr((2, 3, 2), ptr) };
        println!("array is {:?}", array);

    }
}
