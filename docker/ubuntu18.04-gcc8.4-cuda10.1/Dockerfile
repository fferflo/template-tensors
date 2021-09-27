FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# Add cmake repository: https://apt.kitware.com/
RUN apt-get update -y \
 && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget \
 && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
 && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

# Install dependencies
RUN apt-get update -y && apt-get install -y build-essential gcc-8 g++-8 git pcl-tools imagemagick cmake python3 python3-pip libboost-all-dev libomp-dev \
        libopencv-dev libsoil-dev libfreeimage-dev freeglut3-dev libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev libflann-dev \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10

# Tensorflow version >= 2 is not available
# RUN pip3 install tensorflow-gpu
RUN pip3 install cupy

RUN git clone https://gitlab.com/libeigen/eigen.git && mkdir eigen/build && cd eigen/build && cmake .. && make -j8 && make install -j8 && cd ../.. && rm -rf eigen
RUN git clone https://github.com/rogersce/cnpy && cd cnpy && cmake . && make -j8 && make install -j8 && cd .. && rm -rf cnpy
RUN git clone https://github.com/RainerKuemmerle/g2o && cd g2o && cmake -DBUILD_WITH_MARCH_NATIVE=ON . && make -j8 && make install -j8 && cd .. && rm -rf g2o
RUN git clone https://github.com/ddiakopoulos/tinyply && cd tinyply && cmake -DSHARED_LIB=ON . && make -j8 && make install -j8 && cd .. && rm -rf tinyply
RUN git clone https://github.com/dmlc/dlpack && cd dlpack && cmake . && make -j8 && make install -j8 && cd .. && rm -rf dlpack
RUN git clone https://github.com/USCiLab/cereal && cd cereal && cmake -DJUST_INSTALL_CEREAL=ON . && make -j8 && make install -j8 && cd .. && rm -rf cereal
# apt install libtbb-dev does not have a cmake config, github tbb has error: 'task_scheduler_observer' has not been declared
# RUN git clone https://github.com/oneapi-src/oneTBB && cd oneTBB && cmake -DTBB_STRICT=OFF . && make -j8 && make install -j8 && cd .. && rm -rf oneTBB
RUN git clone https://github.com/PointCloudLibrary/pcl && cd pcl && cmake . && make -j8 && make install -j8 && cd .. && rm -rf pcl

# Tensorflow c-api is missing TFE_HandleFromDLPack
#RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz \
# && tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz \
# && ldconfig
#RUN git clone https://github.com/fferflo/cppflow && cd cppflow && git checkout -t origin/add_cmake && cmake -DBUILD_EXAMPLES=OFF . && make -j8 && make install -j8 && cd .. && rm -rf cppflow

RUN apt-get autoclean && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*
