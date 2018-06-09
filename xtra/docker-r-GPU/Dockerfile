FROM nvidia/cuda:9.1-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties apt-transport-https

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    add-apt-repository 'deb [arch=amd64] https://cran.rstudio.com/bin/linux/ubuntu xenial/' && \
    apt-get update && \
    apt-get install -y r-base

RUN apt-get install -y git wget libcurl4-openssl-dev cmake


ENV MAKE="make -j$(nproc)"

RUN R -e 'install.packages(c("ROCR","data.table","R6","jsonlite"), repos = c(CRAN = "https://cran.rstudio.com/"))'


RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make install -j


RUN apt-get install -y libboost-dev libboost-system-dev libboost-filesystem-dev

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN git clone --recursive https://github.com/Microsoft/LightGBM && \
    cd LightGBM/R-package && \
    sed -i 's/use_gpu <- FALSE/use_gpu <- TRUE/' src/install.libs.R && \
    sed -i 's/DUSE_GPU=ON/DUSE_GPU=1 -DOpenCL_LIBRARY=\/usr\/local\/cuda\/lib64\/libOpenCL.so -DOpenCL_INCLUDE_DIR=\/usr\/local\/cuda\/include\//' src/install.libs.R&& \
    R CMD INSTALL --build . --no-multiarch


RUN  apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["R","--vanilla"]

