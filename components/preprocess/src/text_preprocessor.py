from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text

class TextPreprocessor:
    def __init__(self, max_sequence_length):
        self._max_sequence_length = max_sequence_length
        self._labels = None
        self.number_words = None
        self._tokenizer = None

    def fit(self, instances):
        tokenizer = text.Tokenizer(lower=False, filters=[], oov_token=None)
        tokenizer.fit_on_texts(instances)
        self._tokenizer = tokenizer
        self.number_words = len(tokenizer.word_index) + 1

    def transform(self, instances):
        sequences = self._tokenizer.texts_to_sequences(instances)
        padded = pad_sequences(sequences, maxlen=self._max_sequence_length, padding='post')
        return padded
