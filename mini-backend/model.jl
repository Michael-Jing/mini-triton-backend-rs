
include("types.jl")
function f(x,y)
    x + y
end

function mydot(a, b)
    open("output.txt", "w") do file
        write(file, a)
    end
    a * b
end

function execute(request::InferRequest)
    request_id = request.request_id
    correlation_id = request.correlation_id
    model_name = request.model_name
    model_version = request.model_version
    output_name = request.requested_output_names[0]
    input = request.inputs[0]
    response = InferResponse(input)
    return response
end
