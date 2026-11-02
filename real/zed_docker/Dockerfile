# =============================================================================
# ZED SDK container — lets a Fedora host talk to a Stereolabs ZED camera
# without fighting glibc/CUDA mismatches from Stereolabs' Ubuntu-only .run
# installer. Base image already contains the ZED SDK + matching CUDA.
#
# Tag verified against this host on 2026-08-28:
#   - NVIDIA driver 610.57.04 (akmod-nvidia, RPM Fusion), max supported
#     CUDA 13.3 per `nvidia-smi`
#   - nvidia-container-toolkit 1.20.0, `nvidia` runtime registered in
#     `docker info` (Docker 29.7.2 / Compose v5.5.0)
#   - `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04
#     nvidia-smi` confirmed the GPU is visible in a container
# CUDA_VERSION just needs to stay <= the driver's max (13.3 here) — this
# driver has headroom for the whole stereolabs/zed CUDA 11.8-13.3 range, so
# CUDA_VERSION was left at 12.1 (widest tested compatibility) rather than
# bumped. If you rebuild on a different machine, re-check with `nvidia-smi`
# and browse tags at https://hub.docker.com/r/stereolabs/zed/tags — a driver
# older than CUDA 12.1 needs CUDA_VERSION dropped to match.
# =============================================================================
ARG ZED_SDK_VERSION=4.2
ARG CUDA_VERSION=12.1
ARG UBUNTU_VERSION=22.04

FROM stereolabs/zed:${ZED_SDK_VERSION}-devel-cuda${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}

# libgl1/libglib/libgtk2/libsm6/libxext6/libxrender1 -> needed for OpenCV's
# GUI backend (cv2.namedWindow/imshow) at runtime; the stereolabs base image
# is minimal and doesn't include them. NOTE: these libraries alone are not
# enough -- requirements.txt must install `opencv-python`, not
# `opencv-python-headless` (headless is compiled with zero GUI backend at
# all, so cv2.namedWindow fails with "function not implemented" regardless
# of what's on the system).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libgtk2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Build the pyzed Python bindings against the SDK baked into this image.
# (Ships as get_python_api.py inside every stereolabs/zed image; path is
# stable across recent SDK versions but check `find / -name get_python_api.py`
# inside the container if a future base image moves it.)
RUN python3 /usr/local/zed/get_python_api.py

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# get_python_api.py's own pip resolution leaves dist-packages with BOTH
# numpy 1.26.4 and 2.2.4 dist-info side by side (`import numpy` reports
# 1.26.4, `pip list` reports 2.2.4) — a corrupted mixed install. pyzed's
# prebuilt .whl is compiled against the numpy<2.0 ABI it explicitly pins,
# so under that mix Mat.get_data() hands back something cv2 doesn't
# recognize as an ndarray (`cvtColor: src is not a numpy array`). Force a
# clean single reinstall, last, so nothing after this touches numpy again.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps "numpy<2.0,>=1.13"

# Same corruption class, different package: `rembg` (in requirements.txt for
# the ML stack) pulls in an unpinned `opencv-python-headless` as its own
# transitive dependency, which lands in dist-packages ALONGSIDE the GUI
# `opencv-python` from earlier -- pip treats them as unrelated packages
# since they're differently named, so it installs both instead of resolving
# a conflict, and whichever wrote its `cv2/` files last wins at import time
# (silently: `cv2.getBuildInformation()` goes from GUI:QT5 to GUI:NONE).
# Force a clean single reinstall, last, so nothing installed above can
# quietly swap the GUI build out again.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps opencv-python==4.11.0.86

ENV PYTHONUNBUFFERED=1
# HumanEgo/ gets bind-mounted at /workspace/HumanEgo (see docker-compose.yml)
# so run_inference.py's flat imports (CamZED, utils.*, preprocess.*) resolve.
ENV PYTHONPATH=/workspace/HumanEgo
WORKDIR /workspace/HumanEgo

CMD ["/bin/bash"]
