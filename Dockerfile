FROM ubuntu:24.04

ENV LANG=C.UTF-8

# install requirements
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
  git \
  patch \
  clang-17 \
  llvm-17 \
  cmake \
  curl \
  lld \
  ninja-build \
  python3.8 python3.8-venv python3.8-dev

  # copy libs
  COPY /lib /home/lib
  
  # Set up Python environment
  ENV VIRTUAL_ENV=/home/.venv
  RUN python3.8 -m venv $VIRTUAL_ENV
  ENV PATH="$VIRTUAL_ENV/bin:$PATH"
  
  # Install Tensorflow
  ENV CC=clang-17
  ENV CXX=clang-17++
  RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-arm64 -o /usr/local/bin/bazelisk && \
    chmod +x /usr/local/bin/bazelisk
  WORKDIR /home/lib/tensorflow
  RUN ./configure
  # ENV BAZEL_OPTS=" --host_jvm_args=-Xms1024m --host_jvm_args=-Xmx2048m"
  RUN bazelisk build --jvmopt="-Xms8192m" --jvmopt="-Xmx16384m" //tensorflow/compiler/mlir/lite:flatbuffer_translate
  RUN bazelisk build --local_ram_resources=HOST_RAM*.8 --jvmopt="-Xms8192m" --jvmopt="-Xmx16384m"  //tensorflow/compiler/mlir:tf-opt

  RUN apt-get install -y libomp-dev
  # Ensure Clang finds the OpenMP headers and libraries
  RUN ln -s /usr/lib/llvm-17/include/omp.h /usr/local/include/omp.h && \
    ln -s /usr/lib/llvm-17/lib/libomp.so /usr/local/lib/libomp.so

  # Install Baco
  ENV CC=gcc
  ENV CXX=g++
  RUN pip install --upgrade pip
  WORKDIR /home/lib/baco
  RUN pip install -e /home/lib/baco
  

  # MLIR Python bindings prerequisites
  RUN pip install -r /home/lib/llvm-project/mlir/python/requirements.txt
  
  # Compile LLVM
  ENV CC=clang-17
  ENV CXX=clang-++17
  WORKDIR /home/lib/llvm-project
  RUN patch /home/patches/dump_transform_script_from_mlir.patch
  RUN cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='mlir' -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DLLVM_PARALLEL_COMPILE_JOBS=10 -DLLVM_PARALLEL_LINK_JOBS=10 -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE='$VIRTUAL_ENV/bin/python'
  RUN cmake --build build --target mlir-opt
  RUN cmake --build build --target mlir-transform-opt
  
  # Install zsh
  RUN apt-get update && \
  apt-get install -y zsh wget git curl
  RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)" "" --unattended
  
  # Set zsh as the default shell
  RUN chsh -s $(which zsh)
  
  
  RUN apt-get install -y pkg-config libhdf5-dev
  RUN pip install optimum[exporters-tf] tf-keras
  
  # Copy data
  COPY /patches /home/patches
  COPY /scripts /home/scripts
  RUN mkdir /home/models
  

