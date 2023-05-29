#!/bin/bash

APP_NAME=knowledgeable-prompt-tuning
VERSION_CODE=0.1.0
IMAGE_NAME=$APP_NAME:$VERSION_CODE

echo
echo Using version code: $VERSION_CODE
echo Image name: $IMAGE_NAME
echo

docker build . -t $IMAGE_NAME

echo
echo Run command to push image to docker registry:
echo "    docker push $IMAGE_NAME"-
echo
