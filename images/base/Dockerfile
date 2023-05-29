# 使用基于Ubuntu 20.04的基础镜像
FROM ubuntu:20.04 as build

# 作者和维护者信息（可选）
LABEL maintainer="Moyu <Weihp16@gmail.com>"

# 更新系统，并安装必要的软件包
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
            build-essential \
            curl \
            ca-certificates \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            wget \
            libffi-dev \
            liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python 3.11.3
RUN curl -O https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tar.xz \
    && tar -xvf Python-3.11.3.tar.xz \
    && cd Python-3.11.3 \
    && ./configure --enable-optimizations --prefix=/opt/python \
    && make -j "$(nproc)" \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.11.3 \
    && rm -f Python-3.11.3.tar.xz

# 使用一个干净的Ubuntu 20.04基础镜像，以减少最终镜像的大小
FROM ubuntu:20.04

# 安装运行时需求的软件包
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl1.1 \
        zlib1g \
        libbz2-1.0 \
        libreadline8 \
        libsqlite3-0 \
        wget \
        libffi7 \
        liblzma5 \
    && rm -rf /var/lib/apt/lists/*

# 复制已经构建好的Python 3.11.3到新的基础镜像中
COPY --from=build /opt/python /opt/python

# 将/opt/python/bin添加到环境变量中，以便能够运行python和pip
ENV PATH="/opt/python/bin:${PATH}"

# 设置工作目录（可选）
WORKDIR /app

# 将python3.11设置为默认的python版本
RUN ln -s /opt/python/bin/python3.11 /usr/local/bin/python \
    && ln -s /opt/python/bin/pip3.11 /usr/local/bin/pip