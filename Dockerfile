FROM ubuntu:24.04

ENV LANG=C.UTF-8

# install requirements
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
  git \
  patch \
  pkg-config \
  clang-18 \
  cmake \
  curl \
  libblas-dev \
  libhdf5-dev \
  libomp-dev \
  lld \
  llvm-18 \
  ninja-build \
  python3.8 python3.8-venv python3.8-dev \
  python3.9 python3.9-venv python3.9-dev \
  vim

# copy libs
COPY /lib/baco /home/lib/baco
COPY /lib/llvm-project /home/lib/llvm-project
COPY /lib/tensorflow /home/lib/tensorflow
COPY /lib/libxsmm /home/lib/libxsmm

COPY /patches /home/patches

# Create symlinks for clang and clang++
RUN ln -s /usr/bin/clang-18 /usr/bin/clang && \
ln -s /usr/bin/clang++-18 /usr/bin/clang++

# Install Bazelisk
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-arm64 -o /usr/local/bin/bazelisk && \
chmod +x /usr/local/bin/bazelisk

# Set up Python3.9 environment for Enzyme only
ENV VIRTUAL_ENV39=/home/.venv39
RUN python3.9 -m venv $VIRTUAL_ENV39

# Install Enzyme
ENV CC=clang-18
ENV CXX=clang-18++
WORKDIR /home/lib/Enzyme-JaX
RUN git clone https://github.com/EnzymeAD/Enzyme-JaX /home/lib/Enzyme-JaX
RUN git checkout fd3d89f57661a11299e31d61cbefdae959bb2599
RUN . $VIRTUAL_ENV39/bin/activate && HERMETIC_PYTHON_VERSION=3.9 bazelisk build -c opt :enzyme_ad
RUN . $VIRTUAL_ENV39/bin/activate && python3.9 -m pip install --upgrade --force-reinstall bazel-bin/enzyme_ad*.whl
RUN . $VIRTUAL_ENV39/bin/activate && python3.9 test/llama.py

# Set up general Python environment
ENV VIRTUAL_ENV=/home/.venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install optimum[exporters-tf] tf-keras

# Install Baco
RUN pip install --upgrade pip
WORKDIR /home/lib/baco
RUN pip install -e /home/lib/baco

# Install libxsmm
WORKDIR /home/lib/libxsmm
RUN make -j
ENV LIBXSMM_DIR=/home/lib/libxsmm
ENV LD_LIBRARY_PATH=$LIBXSMM_DIR/lib:$LD_LIBRARY_PATH

# MLIR Python bindings prerequisites
RUN pip install -r /home/lib/llvm-project/mlir/python/requirements.txt

# Compile MLIR
WORKDIR /home/lib/llvm-project
RUN patch /home/patches/dump_transform_script_from_mlir.patch
RUN patch -p1 < /home/patches/timing_mlir_opt.patch
RUN patch -p1 < /home/patches/timing_transforms.patch
RUN cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='mlir' -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DLLVM_PARALLEL_COMPILE_JOBS=10 -DLLVM_PARALLEL_LINK_JOBS=10 -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE='$VIRTUAL_ENV/bin/python'
RUN cmake --build build --target mlir-opt
RUN cmake --build build --target mlir-transform-opt
RUN cmake --build build --target mlir-translate
RUN cmake --build build --target mlir_runner_utils
RUN cmake --build build --target mlir_c_runner_utils

# Build Tensorflow
WORKDIR /home/lib/tensorflow
RUN ./configure
RUN bazelisk build --jvmopt="-Xms8192m" --jvmopt="-Xmx16384m" //tensorflow/compiler/mlir/lite:flatbuffer_translate
RUN bazelisk build --jvmopt="-Xms8192m" --jvmopt="-Xmx16384m"  //tensorflow/compiler/mlir:tf-opt

# Prepare performance exploration
COPY /lib/Performance_Exploration /home/lib/Performance_Exploration

WORKDIR /home/lib/Performance_Exploration
RUN pip install -r requirements.txt
WORKDIR /home/lib/Performance_Exploration/build
RUN cmake .. -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
RUN make batch_matmul
RUN mkdir /home/bin && cp /home/lib/Performance_Exploration/build/batch_matmul /home/bin/batch_matmul

# Install zsh
RUN apt-get update && \
apt-get install -y zsh wget git curl
RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)" "" --unattended

# Set zsh as the default shell
RUN chsh -s $(which zsh)

# Copy data
COPY /scripts /home/scripts
RUN chmod u+x /home/scripts/plot_performance_evolution.sh
RUN chmod u+x /home/scripts/show_spinner.sh
RUN mv /home/scripts/run_all.sh /home/run_all.sh
RUN chmod u+x /home/run_all.sh
RUN mkdir /home/models
RUN mkdir /home/results
RUN mkdir /home/results/logs

COPY README.md /home/README.md
