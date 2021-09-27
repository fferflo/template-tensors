FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Add cmake repository: https://apt.kitware.com/
RUN apt-get update -y \
 && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget \
 && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
 && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'

# Install dependencies
RUN apt-get update -y && apt-get install -y build-essential git imagemagick cmake python3 python3-pip libboost1.71-all-dev libomp-dev \
        libeigen3-dev libopencv-dev libtbb-dev libsoil-dev libfreeimage-dev freeglut3-dev libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev
# libpcl-dev is buggy: pcl::io::loadPCDFile causes SIGSEGV when compiling with nvcc # pcl-tools

# Tensorflow version according to https://www.tensorflow.org/install/source#gpu
RUN pip3 install tensorflow-gpu==2.4
# Upgrading Cython to a version compatible with cupy
RUN pip3 install --upgrade Cython
RUN pip3 install cupy

RUN git clone https://github.com/rogersce/cnpy && cd cnpy && cmake . && make -j8 && make install -j8 && cd .. && rm -rf cnpy
RUN git clone https://github.com/RainerKuemmerle/g2o && cd g2o && cmake -DBUILD_WITH_MARCH_NATIVE=ON . && make -j8 && make install -j8 && cd .. && rm -rf g2o
RUN git clone https://github.com/ddiakopoulos/tinyply && cd tinyply && cmake -DSHARED_LIB=ON . && make -j8 && make install -j8 && cd .. && rm -rf tinyply
RUN git clone https://github.com/dmlc/dlpack && cd dlpack && cmake . && make -j8 && make install -j8 && cd .. && rm -rf dlpack
RUN git clone https://github.com/USCiLab/cereal && cd cereal && cmake -DJUST_INSTALL_CEREAL=ON . && make -j8 && make install -j8 && cd .. && rm -rf cereal

# Tensorflow c-api is missing TFE_HandleFromDLPack
#RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz \
# && tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz \
# && ldconfig
#RUN git clone https://github.com/fferflo/cppflow && cd cppflow && git checkout -t origin/add_cmake && cmake -DBUILD_EXAMPLES=OFF . && make -j8 && make install -j8 && cd .. && rm -rf cppflow

RUN apt-get autoclean && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*
