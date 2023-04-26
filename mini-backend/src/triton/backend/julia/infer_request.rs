use std::collections::HashSet;

use jlrs::data::types::foreign_type::{OpaqueType, ForeignType};

use super::jb_tensor::JbTensor;

pub struct InferRequest {
    request_id_: String ,
    correlation_id_: u64 ,
    inputs_: Vec<JbTensor>,
    requested_output_names_: HashSet<String>,
    model_name_: String,
    model_version_: i64,
    flags_: u32,
    // response_factory_addresses_: std::core::ptr,
    // request_address_: std::ptr,
}
impl InferRequest {
    pub fn Inputs(self) -> Vec<JbTensor> {
        return self.inputs_;
    }
    pub fn RequestId(self) -> String{
        return self.request_id_;
    }

    pub fn CorrelationId(self) -> u64 {
        return self.correlation_id_;
    }

    pub fn RequestedOutputNames(self) -> HashSet<String> {
        return self.requested_output_names_;
    }

    pub fn ModelName(self) -> String {
        return self.model_name_;
    }

    pub fn ModelVersion(self) -> i64 {
        return self.model_version_;
    }

    pub fn Flags(self) -> u32 {
        return self.flags_;
    }

}

unsafe impl OpaqueType for InferRequest {} 