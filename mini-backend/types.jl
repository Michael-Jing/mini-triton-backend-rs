import JlrsReflect
struct JbTensor
    name::String
    dtype::UInt32
    dims::Vector{Int64}
    memory_ptr::Union{Array{Real, 1}, Array{Real, 2}, Array{Real, 3}, Array{Real, 4}}
    memory_type_id::Int64
    memory_type::UInt32
    byte_size::UInt64
end


struct JbError
    message::String
end
struct InferResponse
    output_tensors::Vector{JbTensor}
    error::JbError
end

struct InferRequest 
    request_id::String
    correlation_id::Int32
    model_name::String
    model_version::Int32
    flags::Int32
    inputs::Vector{JbTensor}
    requested_output_names::Vector{String}
end


wrappers = JlrsReflect.reflect([JbTensor, JbError, InferRequest, InferResponse ]);

# Print wrappers to standard output
println(wrappers)

# Write wrappers to file
open("julia_wrappers2.rs", "w") do f
   write(f, wrappers)
end