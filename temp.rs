pub struct BackendInputCollector {}
pub struct InputIterator {
    /*TRITONBACKEND_Request** requests_;
    const uint32_t request_count_;
    std::vector<TRITONBACKEND_Response*>* responses_;
    const char* input_name_;
    const char* host_policy_;
    const bool coalesce_request_input_;

    TRITONBACKEND_Input* curr_input_;
    size_t curr_request_idx_;
    size_t curr_buffer_idx_;
    uint32_t curr_buffer_cnt_;
    bool reach_end_; */
    requests_: *mut *mut TRITONBACKEND_Request,
    request_count_: u32,
    responses_: Vec<*mut TRITONBACKEND_Response>,
    input_name_: *const i8,
    host_policy_: *const i8,
    coalesce_request_input_: bool,
    curr_input_: *mut TRITONBACKEND_Input,
    curr_request_idx_: usize,
    curr_buffer_idx_: u32,
    curr_buffer_cnt_: u32,
    reach_end_: bool,
}

pub struct MemoryDesc {
    /*const char* buffer_;
    size_t byte_size_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t memory_type_id_; */
    buffer_: *mut *const char,
    memory_type_id_: i64,
    memory_type_: TRITONSERVER_MemoryType,
    byte_size_: u64,
}
pub struct ContiguousBuffer {
    /*  MemoryDesc memory_desc_;
    size_t start_request_idx_;
    size_t end_request_idx_; */
    start_request_idx_: usize,
    end_request_idx_: usize,
    memory_desc_: MemoryDesc,
}
impl InputIterator {
    pub fn GetNextContiguousInput(&mut self, input: *mut ContiguousBuffer) -> bool {
        if self.reach_end_ || self.curr_buffer_idx_ >= self.curr_buffer_cnt_ {
            return false;
        }
        unsafe {
            TRITONBACKEND_InputBufferForHostPolicy(
                self.curr_input_,
                self.host_policy_,
                self.curr_buffer_idx_,
                (*input).memory_desc_.buffer_ as *mut *const ffi::c_void,
                &mut (*input).memory_desc_.byte_size_,
                &mut (*input).memory_desc_.memory_type_,
                &mut (*input).memory_desc_.memory_type_id_,
            );
        }
        self.curr_buffer_idx_ += 1;
        // start and end request idx?
        unsafe {
            (*input).start_request_idx_ = self.curr_request_idx_;
            (*input).end_request_idx_ = self.curr_request_idx_;
        }

        if !self.coalesce_request_input_ {
            if (self.curr_buffer_idx_ >= self.curr_buffer_cnt_) {
                // done with one request, setup the next, if not done,
                // just return, which indicates that only one buffer is processed
                // move to next request
                self.curr_request_idx_ += 1;
                if (self.curr_request_idx_ < self.request_count_) {
                    let response = self.responses_[self.curr_request_idx_];
                    // set curr_input_, curr_buffer_idx_ and curr_buffer_cnt to reflect the reality on the new request
                    // if unable to get input, response with error
                    RESPOND_AND_SET_NULL_IF_ERROR(
                        &response,
                        TRITONBACKEND_RequestInput(
                            requests_[curr_request_idx_],
                            input_name_,
                            &curr_input_,
                        ),
                    );
                    // if unable to get InputProperties, response with error
                    RESPOND_AND_SET_NULL_IF_ERROR(
                        &response,
                        TRITONBACKEND_InputPropertiesForHostPolicy(
                            curr_input_,
                            host_policy_,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            &curr_buffer_cnt_,
                        ),
                    );
                    // reset buffer idx, so can start to process buffer of current request in next call
                    curr_buffer_idx_ = 0;
                } else {
                    reach_end_ = true;
                }
            }
            // please come back later
            return true;
        }
        // dummy return  placeholder
        return false;
    }
}
