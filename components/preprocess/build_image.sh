#!/bin/sh

image_name=us-east1-docker.pkg.dev/$PROJECT_ID/ner-repo/preprocess
image_tag=latest

full_image_name=${image_name}:${image_tag}
base_image_tag=2.11.0

cd "$(dirname "$0")" 

docker build --build-arg BASE_IMAGE_TAG=${base_image_tag} -t "${full_image_name}" .
docker push "$full_image_name"
