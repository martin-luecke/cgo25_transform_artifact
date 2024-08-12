FROM ubuntu:24.04

ENV LANG=C.UTF-8

# install requirements
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
  git \
  patch \
  clang \
  cmake \
  lld \
  ninja-build \
  python3.8 python3.8-venv python3.8-dev

# copy libs
COPY /lib /home/lib
COPY /patches /home/patches

# Set up Python environment
ENV VIRTUAL_ENV=/home/.venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Baco
RUN pip install --upgrade pip
WORKDIR /home/lib/baco
RUN pip install -e /home/lib/baco

# MLIR Python bindings prerequisites
RUN pip install -r /home/lib/llvm-project/mlir/python/requirements.txt

# Compile LLVM
WORKDIR /home/lib/llvm-project
RUN patch /home/patches/dump_transform_script_from_mlir.patch
RUN cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='mlir' -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_PARALLEL_COMPILE_JOBS=10 -DLLVM_PARALLEL_LINK_JOBS=10 -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE='$VIRTUAL_ENV/bin/python'
RUN cmake --build build --target mlir-opt
RUN cmake --build build --target mlir-transform-opt


