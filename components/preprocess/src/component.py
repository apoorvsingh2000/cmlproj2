import argparse
import os
from pathlib import Path
import pickle

import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from text_preprocessor import TextPreprocessor

PREPROCESS_FILE = 'processor_state.pkl'

def read_data(path: str) -> pd.DataFrame:
    with tf.io.gfile.GFile(path, 'r') as f:
        print(f'Processing file: {path}')
        return pd.read_csv(f, on_bad_lines='skip')

def save_pickle(obj, path: str) -> None:
    with tf.io.gfile.GFile(path, 'wb') as f:
        pickle.dump(obj, f)

def main():
    parser = argparse.ArgumentParser(description='NER data preprocessing')
    
    parser.add_argument('--input-path', type=str, help='Input CSV file path')
    parser.add_argument('--output-x-path', type=str, help='Output X pickle path')
    parser.add_argument('--output-x-path-file', type=str, help='Path to write X file location')
    parser.add_argument('--output-y-path', type=str, help='Output Y pickle path')
    parser.add_argument('--output-y-path-file', type=str, help='Path to write Y file location')
    parser.add_argument('--output-preprocessing-state-path', type=str, help='Output preprocessor state directory')
    parser.add_argument('--output-preprocessing-state-path-file', type=str, help='Path to write preprocessor state location')
    parser.add_argument('--output-tags', type=str, help='Path to save number of tags')
    parser.add_argument('--output-words', type=str, help='Path to save number of words')

    args = parser.parse_args()

    df = read_data(args.input_path)

    drop_cols = [
        'Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
        'next-next-shape', 'next-next-word', 'next-pos', 'next-shape', 'next-word',
        'prev-iob', 'prev-lemma', 'prev-pos', 'prev-prev-iob', 'prev-prev-lemma',
        'prev-prev-pos', 'prev-prev-shape', 'prev-prev-word', 'prev-shape', 'prev-word',
        'pos', 'shape'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    grouped = df.groupby('sentence_idx').apply(
        lambda s: [(w, t) for w, t in zip(s['word'], s['tag'])]
    )
    sentences = list(grouped)
    texts = [' '.join([w for w, _ in sent]) for sent in sentences]

    maxlen = max(len(sent) for sent in sentences)
    print(f'Max sequence length: {maxlen}')

    words = sorted({w for sent in sentences for w, _ in sent})
    tags = sorted({t for sent in sentences for _, t in sent})
    print(f'Vocab size: {len(words)}, Tag count: {len(tags)}')

    for p in [args.output_x_path, args.output_y_path, args.output_preprocessing_state_path]:
        tf.io.gfile.makedirs(os.path.dirname(p))

    processor = TextPreprocessor(maxlen)
    processor.fit(texts)
    processor.labels = tags
    preprocessor_save_path = os.path.join(args.output_preprocessing_state_path, PREPROCESS_FILE)
    processor.save(preprocessor_save_path)

    X = processor.transform(texts)

    tag2idx = {t: i for i, t in enumerate(tags)}
    y_indices = [[tag2idx[t] for _, t in sent] for sent in sentences]
    y_padded = pad_sequences(y_indices, maxlen=maxlen, padding='post', value=tag2idx.get('O', 0))
    y = [to_categorical(seq, num_classes=len(tags)) for seq in y_padded]

    save_pickle(X, args.output_x_path)
    save_pickle(y, args.output_y_path)

    downstream = [
        (args.output_x_path_file, args.output_x_path),
        (args.output_y_path_file, args.output_y_path),
        (args.output_preprocessing_state_path_file, preprocessor_save_path),
        (args.output_tags, str(len(tags))),
        (args.output_words, str(len(words)))
    ]

    for file_path, content in downstream:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_text(content)

if __name__ == '__main__':
    main()
