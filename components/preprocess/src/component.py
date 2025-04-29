import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_preprocessor import TextPreprocessor

from google.cloud import storage

PREPROCESS_FILE = 'processor_state.pkl'

def upload_to_gcs(local_path, gcs_path):
    """Uploads a local file to GCS"""
    client = storage.Client()
    bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

def read_data(input_path):
    if input_path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_path = input_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        contents = blob.download_as_text()
        from io import StringIO
        data = pd.read_csv(StringIO(contents))
    else:
        data = pd.read_csv(input_path)
    return data

def main():
    parser = argparse.ArgumentParser(description='Preprocessing component')
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-x-path', type=str, required=True)
    parser.add_argument('--output-y-path', type=str, required=True)
    parser.add_argument('--output-preprocessing-state-path', type=str, required=True)
    
    args = parser.parse_args()

    # Read input data
    data = read_data(args.input_path)

    # Preprocessing
    data = data.drop([
        'Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
        'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
        'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
        'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
        'prev-prev-word', 'prev-shape', 'prev-word', "pos", "shape"
    ], axis=1)

    # Build sentences
    grouped = data.groupby("sentence_idx").apply(lambda s: [(w, t) for w, t in zip(s["word"], s["tag"])])
    sentences = [s for s in grouped]
    sentences_list = [" ".join([s[0] for s in sent]) for sent in sentences]

    maxlen = max([len(s) for s in sentences])
    words = list(set(data["word"].values))
    tags = list(set(data["tag"].values))

    processor = TextPreprocessor(140)
    processor.fit(sentences_list)
    processor.labels = tags

    X = processor.transform(sentences_list)

    tag2idx = {t: i for i, t in enumerate(tags)}
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=len(tags)) for i in y]

    # Save X locally
    x_local = "/tmp/X.pkl"
    with open(x_local, "wb") as f:
        pickle.dump(X, f)
    upload_to_gcs(x_local, args.output_x_path)

    # Save y locally
    y_local = "/tmp/y.pkl"
    with open(y_local, "wb") as f:
        pickle.dump(y, f)
    upload_to_gcs(y_local, args.output_y_path)

    # Save processor locally
    processor_local = "/tmp/" + PREPROCESS_FILE
    with open(processor_local, "wb") as f:
        pickle.dump(processor, f)
    upload_to_gcs(processor_local, os.path.join(args.output_preprocessing_state_path, PREPROCESS_FILE))

if __name__ == "__main__":
    main()
