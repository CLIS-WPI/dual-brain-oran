# Harmony TaaS — Custom OAI gNB with UHD 4.10.0.0
#
# This Dockerfile extends the official OAI gNB image with an upgraded UHD
# version (4.10.0.0) from the Ettus Research GitHub repository.
#
# UHD 4.10.0.0 adds support for:
# - USRP X420 (new hardware)
# - Timed complex gain feature
# - Improved DPDK support
# - Better resource utilization
# - Bug fixes for E31x/E3xx/B2xx/X410 GPIO and tuning
#
# Build with:
#   docker build --build-arg GNB_BASE_IMAGE=oaisoftwarealliance/oai-gnb:2026.w27 \
#     -t localhost/harmony/oai-gnb-uhd410:2026.w27 \
#     -f docker/custom-gnb/Dockerfile .

ARG GNB_BASE_IMAGE=docker.io/oaisoftwarealliance/oai-gnb:2026.w27
FROM ${GNB_BASE_IMAGE}

# UHD 4.10.0.0 release tag from GitHub
ARG UHD_VERSION=v4.10.0.0

# Install UHD build dependencies (required for building from source)
# UHD 4.10.0.0 requires Boost >= 1.71, CMake >= 3.16, Python 3, etc.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-all-dev \
        libusb-1.0-0-dev \
        libudev-dev \
        python3-dev \
        python3-mako \
        python3-numpy \
        python3-requests \
        python3-setuptools \
        doxygen \
        dpdk-dev \
        libdpdk-dev \
        ca-certificates \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Download UHD 4.10.0.0 from Ettus Research GitHub
# (GitLab EureCom is not accessible in this environment, but GitHub is)
WORKDIR /tmp/uhd-build
RUN git clone --depth 1 --branch ${UHD_VERSION} https://github.com/EttusResearch/uhd.git && \
    cd uhd/host && \
    mkdir build && \
    cd build && \
    cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DENABLE_TESTS=OFF \
        -DENABLE_C_API=ON \
        -DENABLE_PYTHON_API=ON \
        -DENABLE_DPDK=ON \
        -DENABLE_USB=ON \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && \
    rm -rf /tmp/uhd-build

# Download and install UHD FPGA images for the target USRP hardware
# (required for USRP N310, B210, X310, etc.)
RUN uhd_images_downloader -t n3xx -t b2xx -t x3xx

# Verify UHD installation
RUN uhd_find_devices --help && \
    ldconfig -p | grep libuhd && \
    echo "UHD 4.10.0.0 installation verified"

# Restore original working directory and entrypoint
WORKDIR /opt/oai-gnb
# The base image's ENTRYPOINT and CMD are preserved

# Add label to track UHD version for inventory purposes
LABEL uhd.version="4.10.0.0" \
      uhd.source="https://github.com/EttusResearch/uhd/releases/tag/v4.10.0.0" \
      harmony.custom="true" \
      harmony.description="OAI gNB with upgraded UHD 4.10.0.0 for USRP N310"
