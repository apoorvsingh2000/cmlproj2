#!/bin/sh

image_name=us-east1-docker.pkg.dev/$PROJECT_ID/ner-repo/deploy
image_tag=latest

full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"

docker build -t "${full_image_name}" .
docker push "$full_image_name"
