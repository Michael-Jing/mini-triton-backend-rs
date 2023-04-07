import sys
import re
keywords = [
"TRITONBACKEND_Backend",
"TRITONBACKEND_BackendAttribute",
"TRITONBACKEND_Batcher",
"TRITONBACKEND_Input",
"TRITONBACKEND_MemoryManager",
"TRITONBACKEND_Model",
"TRITONBACKEND_ModelInstance",
"TRITONBACKEND_Output",
"TRITONBACKEND_Request",
"TRITONBACKEND_Response",
"TRITONBACKEND_ResponseFactory",
"TRITONBACKEND_State",
"TRITONSERVER_BufferAttributes",
"TRITONSERVER_Error",
"TRITONSERVER_InferenceRequest",
"TRITONSERVER_InferenceResponse",
"TRITONSERVER_InferenceTrace",
"TRITONSERVER_Message",
"TRITONSERVER_Metric",
"TRITONSERVER_MetricFamily",
"TRITONSERVER_Metrics",
"TRITONSERVER_Parameter",
"TRITONSERVER_ResponseAllocator",
"TRITONSERVER_Server",
"TRITONSERVER_ServerOptions"
]


def replace(file_in, file_out):
   

    with open(file_in) as fi, open(file_out, 'w') as fo:
        for line in fi:
            for keyword in keywords:
                line = re.sub(re.escape(f"{keyword}*"), f"struct {keyword}*", line)
            fo.write(line)

def main():
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    print(f"file in is {file_in}")
    print(f"file out is {file_out}")
    replace(file_in, file_out)


if __name__ == "__main__":
    main()


