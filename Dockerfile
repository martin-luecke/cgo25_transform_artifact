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
  python3.9 python3.9-venv python3.9-dev apt-transport-https curl gnupg

RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" |  tee /etc/apt/sources.list.d/bazel.list
RUN apt update && apt install bazel-6.2.1 -y

RUN git clone https://github.com/EnzymeAD/Enzyme-JaX 
WORKDIR /Enzyme-JaX
RUN git checkout fd3d89f57661a11299e31d61cbefdae959bb2599
RUN HERMETIC_PYTHON_VERSION=3.9 bazel-6.2.1 build -c opt :enzyme_ad

# Set up Python environment
ENV VIRTUAL_ENV=/home/.venv
RUN python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip

RUN python3.9 -m pip install --upgrade --force-reinstall bazel-bin/enzyme_ad*.whl
RUN python3.9 test/llama.py


