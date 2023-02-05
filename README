Minimal Nvidia Triton Inference Server Backend written in rust, work in progress
# Get Started

1. build images and start docker
	`docker compose up -d`

2. go inside docker and start triton server
```
	docker exec -it <container name> /bin/bash
	
```

then inside docker
3. cargo build

4. cp target/debug/libtriton_minimal.so /opt/tritonserver/backends/minimal/

5.
`
	tritonserver --model-repository=/models
`
6. from host machine, run minimal_client https://github.com/triton-inference-server/backend/blob/main/examples/clients/minimal_client

