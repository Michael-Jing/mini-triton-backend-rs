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
