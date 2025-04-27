#!/bin/sh

BUCKET="my-kubeflow-bucket-1745602479"

echo "\nCopy component specifications to Google Cloud Storage"

# Copy Preprocess Component
gsutil cp preprocess/component.yaml gs://${BUCKET}/components/preprocess/component.yaml

# Copy Train Component
gsutil cp train/component.yaml gs://${BUCKET}/components/train/component.yaml

# Copy Deploy Component
gsutil cp deploy/component.yaml gs://${BUCKET}/components/deploy/component.yaml
