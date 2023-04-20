use super::{jb_tensor::JbTensor, jb_error::JbError};

pub struct InferResponse {
    output_tensors_: Vec<JbTensor>,
    error_: JbError,
    // gpu_output_buffers_: Vec<()>,
    
    // response_factory_addresses_: std::core::ptr,
    // request_address_: std::ptr,
}