import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
from google.cloud import storage
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, LSTM, Bidirectional, TimeDistributed, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

MODEL_FILE = 'keras_saved_model.h5'

def download_from_gcs(gcs_path, local_path):
    client = storage.Client()
    bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)

def upload_to_gcs(local_path, gcs_path):
    client = storage.Client()
    bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

def load_pickle(path):
    if path.startswith("gs://"):
        local_tmp = "/tmp/tmp_pickle"
        download_from_gcs(path, local_tmp)
        with open(local_tmp, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-x-path', type=str, required=True)
    parser.add_argument('--input-y-path', type=str, required=True)
    parser.add_argument('--input-job-dir', type=str, required=True)
    parser.add_argument('--input-tags', type=int, required=True)
    parser.add_argument('--input-words', type=int, required=True)
    parser.add_argument('--input-dropout', type=float, required=True)
    parser.add_argument('--output-model-path', type=str, required=True)
    parser.add_argument('--output-model-path-file', type=str, required=True)
    args = parser.parse_args()

    X = load_pickle(args.input_x_path)
    y = load_pickle(args.input_y_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # TensorBoard callback
    tensorboard = TensorBoard(log_dir=os.path.join(args.input_job_dir, 'logs'))
    callbacks = [tensorboard]

    # Build Model
    model_input = Input(shape=(140,))
    model = Embedding(input_dim=args.input_words, output_dim=140, input_length=140)(model_input)
    model = Dropout(args.input_dropout)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(args.input_tags, activation="softmax"))(model)
    model = Model(model_input, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    # Train model
    history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=1,
                        validation_split=0.1, verbose=1, callbacks=callbacks)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, np.array(y_test))

    # Save model locally
    model.save(MODEL_FILE)

    # Upload model to GCS
    upload_to_gcs(MODEL_FILE, os.path.join(args.output_model_path, MODEL_FILE))

    # Save metrics for pipeline
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue': accuracy,
            'format': "PERCENTAGE",
        }]
    }
    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Save metadata for TensorBoard visualization
    metadata = {
        'outputs': [{
            'type': 'tensorboard',
            'source': args.input_job_dir,
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    # Save output model path to file
    Path(args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_path_file).write_text(args.output_model_path)

if __name__ == "__main__":
    main()
