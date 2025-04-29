#!/bin/bash
set -e

mkdir -p specs
cp preprocess/component.yaml specs/preprocess.yaml
cp train/component.yaml specs/train.yaml
cp deploy/component.yaml specs/deploy.yaml

echo "Component specifications copied to specs/ folder."
