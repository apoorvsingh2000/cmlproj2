#!/bin/bash
set -e

echo "Building Preprocess component..."
cd preprocess
bash build_image.sh
cd ..

echo "Building Train component..."
cd train
bash build_image.sh
cd ..

echo "Building Deploy component..."
cd deploy
bash build_image.sh
cd ..

echo "All components built and pushed successfully!"
