FROM gurobi/optimizer:10.0.0

ARG http_proxy
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt list --upgradable && \
    apt install -y mercurial vim git gcc gfortran wget make man build-essential libgmp3-dev libboost-all-dev libz-dev liblapack-dev libblas-dev cmake g++ m4 xz-utils libgmp-dev unzip zlib1g-dev libboost-program-options-dev libboost-serialization-dev libboost-regex-dev libboost-iostreams-dev libtbb-dev libreadline-dev pkg-config liblapack-dev libgsl-dev flex bison libcliquer-dev file dpkg-dev libopenblas-dev rpm

ENV APP_PATH=/opt
ENV SCIP_VER=7.0.2
ENV SCIPSDP_VER=3.2.0

WORKDIR $APP_PATH

# install mosek
RUN mkdir -p $APP_PATH/mosek
RUN cd $APP_PATH/mosek && \ 
    wget https://download.mosek.com/stable/9.2.40/mosektoolslinux64x86.tar.bz2 && \
    tar -jxvf mosektoolslinux64x86.tar.bz2
RUN mkdir -p /root/mosek
ENV PATH="$PATH:$APP_PATH/mosek/mosek/9.2/tools/platform/linux64x86/bin"

# install scip
RUN mkdir -p $APP_PATH/scip
RUN cd $APP_PATH/scip && \
    wget https://www.scipopt.org/download/release/scipoptsuite-$SCIP_VER.tgz && \
    tar xzf scipoptsuite-$SCIP_VER.tgz && \
    cd scipoptsuite-$SCIP_VER && \
    make ZLIB=false READLINE=false GM=false 

# install scipsdp
RUN mkdir -p $APP_PATH/scipsdp
RUN cd $APP_PATH/scipsdp && \
    wget http://www.opt.tu-darmstadt.de/scipsdp/downloads/scipsdp-$SCIPSDP_VER.tgz && \
    tar xzf scipsdp-$SCIPSDP_VER.tgz && \
    cd scipsdp-$SCIPSDP_VER && \
    mkdir -p lib &&\
    cd lib &&\
    ln -s $APP_PATH/scip/scipoptsuite-$SCIP_VER/scip/ scip&&\
    mkdir -p include &&\
    mkdir -p static &&\
    mkdir -p shared &&\
    ln -s $APP_PATH/mosek/mosek/9.2/tools/platform/linux64x86/h/ include/mosekh &&\
    ln -s $APP_PATH/mosek/mosek/9.2/tools/platform/linux64x86/bin/libmosek64.so shared/libmosek64.so &&\
    cd .. &&\
    make SDPS=msk 
ENV PATH="$PATH:$APP_PATH/scipsdp/scipsdp-$SCIPSDP_VER/bin"

WORKDIR /home
