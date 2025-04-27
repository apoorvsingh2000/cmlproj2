ARG BASE_IMAGE_TAG=2.11.0
FROM tensorflow/tensorflow:${BASE_IMAGE_TAG}

RUN pip install --no-cache-dir pandas scikit-learn

COPY ./src /pipelines/component/src
