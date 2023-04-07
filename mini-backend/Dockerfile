FROM nvcr.io/nvidia/tritonserver:22.11-py3 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y libclang-dev rapidjson-dev cmake

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ARG CMAKE_VERSION=3.25.1

# RUN apt-get update && apt-get install -y build-essential libssl-dev wget \
#     && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz \
#     && tar zxvf cmake-${CMAKE_VERSION}.tar.gz \
#     && cd cmake-${CMAKE_VERSION} \
#     && ./bootstrap \
#     && make && make install && cd .. \
#     && rm -rf /cmake-${CMAKE_VERSION} && rm cmake-${CMAKE_VERSION}.tar.gz \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz \
&& tar zxvf julia-1.8.5-linux-x86_64.tar.gz

ENV PATH=$PATH:$HOME/.cargo/bin:/opt/tritonserver/julia-1.8.5

WORKDIR /workspace
