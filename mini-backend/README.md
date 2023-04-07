Minimal Nvidia [Triton Inference Server](https://github.com/triton-inference-server) Backend written in rust, work in progress
# Get Started

1. build images and start docker
```
	docker compose up -d
```

2. go inside docker
```
	docker exec -it <container name> /bin/bash
	
```

3. build the project
```
	cargo build
```


4. copy .so file to triton backend folder
```
	cp target/debug/libtriton_minimal.so /opt/tritonserver/backends/minimal/
```

5. start triton server
```
	tritonserver --model-repository=/models
```
6. from host machine, run minimal_client 
```
	python clients/minimal_client
```

# Issues
2. objects and bytes type are not supported
3. how to get the a pointer to the underlying memory buffer (use jl_array_data)
4. how to get data size in bytes (jl_array_len() * size of element)
5. how to get array shape (maybe keep using jl_array_len() for several times)
6. Julia must permute dims before it passing data to rust
