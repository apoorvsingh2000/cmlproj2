ARG BASE_IMAGE_TAG=2.11.0
FROM tensorflow/tensorflow:${BASE_IMAGE_TAG}

# Install any extra packages you need (example: scikit-learn, pandas if used)
RUN pip install --no-cache-dir pandas scikit-learn

# Copy your source code into the container
COPY ./src /pipelines/component/src
