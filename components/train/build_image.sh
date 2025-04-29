#!/bin/bash
PROJECT_ID=cmlproj2
REGION=us-east1
REPO=ner-repo
IMAGE=train
TAG=latest

docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG
