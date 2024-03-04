FROM nvcr.io/nvidia/pytorch:22.09-py3
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

ARG ISAACGYM="repos/isaacgym"

COPY $ISAACGYM/docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY $ISAACGYM/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/isaacgym


ARG USERNAME=user
ARG UID=1001
RUN useradd  --create-home -m -s /bin/bash -u $UID $USERNAME
USER $USERNAME

# RUN useradd --create-home gymuser
# USER gymuser

# copy gym repo to docker
COPY --chown=$USERNAME $ISAACGYM .

ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# install gym modules
RUN pip install --upgrade pip
RUN cd python && pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# install requrment

COPY docker/requirements.txt /home/$USERNAME/
WORKDIR /home/$USERNAME

RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

#RUN git clone https://github.com/pfnet/pfrl.git
#WORKDIR /home/$USERNAME/pfrl

# patch for issue (https://github.com/pfnet/pfrl/issues/154)
# this patch is commited at https://github.com/pfnet/pfrl/commit/fda5361dc233f237518f413478730a53b6065c20
#COPY docker/patch/* /home/$USERNAME/pfrl/

#RUN pip install -e .

#RUN pip uninstall -y gym
#RUN pip install gym

#RUN patch -u pfrl/experiments/prepare_output_dir.py < prepare_output_dir.py.patch
#RUN patch -u pfrl/experiments/evaluator.py < evaluator.py.patch
#RUN patch -u pfrl/utils/recurrent.py < recurrent.py.patch
#RUN patch -u pfrl/agents/ppo.py < ppo.py.patch