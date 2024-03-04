#!/bin/bash
set -e
set -u

username=$(whoami)
uid=$(id -u)

docker build --network host -t ${username}/isaacgym \
    --build-arg USERNAME=${username} \
    --build-arg UID=${uid} \
    -f docker/Dockerfile \
    .
