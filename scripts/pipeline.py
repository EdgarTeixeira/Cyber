import random
import re
import tarfile
import unicodedata
from enum import Enum
from typing import List


def data_archive(archive: str, encoding: str) -> str:
    corpus: List[str] = []

    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            fhandler = tar.extractfile(item)
            if fhandler is not None:
                content = fhandler.read()\
                                  .decode(encoding)\
                                  .replace("\x92", '')\
                                  .replace("\x93", '')\
                                  .replace("\x94", '')\
                                  .replace("\x96", '')\
                                  .replace("\x97", '')
                corpus.append(content)
                fhandler.close()

    return '\n\n'.join(corpus)


def remove_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != 'Mn')


def remove_punctuations(text: str) -> str:
    # TODO: Should ignore numbers? Like 3.0 or 3,0
    return re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\\\]^_`{|}~“”—‘’']+", " ", text)


def remove_case(text: str) -> str:
    return text.lower()


def flatten_text(text: str) -> str:
    return re.sub("\\s+", " ", text)


def build_ngrams(text: str, size: int) -> List[str]:
    if size <= 0:
        raise ValueError("size must be greater than 0")
    return [text[idx:idx + size] for idx in range(len(text) - size + 1)]


def generate_samples(text: str, sample_size: int, seed: int) -> List[str]:
    if sample_size < 1:
        raise ValueError("sample_size must be greater than 1")

    samples = [""] * sample_size
    prng_state = random.getstate()
    random.seed(seed)

    spaces = list(re.finditer(r"\s+", text))
    num_spaces = len(spaces)

    for i in range(sample_size):
        begin = random.randint(0, num_spaces - 1)
        end = begin + random.randint(1, 15)
        if end >= num_spaces:
            end = num_spaces - 1

        samples[i] = text[spaces[begin].end():spaces[end].start()]

    random.setstate(prng_state)

    return samples


class OOVPolicy(Enum):
    ignore = 0
    error = 1
    encode = 2


class IndexEncoder:
    def __init__(self, oov_policy: OOVPolicy = OOVPolicy.ignore) -> None:
        if oov_policy != OOVPolicy.ignore:
            raise NotImplementedError(
                "The current version only supports the ignore policy")

        self.oov_policy = oov_policy

    def fit(self, X: List[str]) -> None:
        alphabet = set(''.join(X))
        self.mapping_size_ = len(alphabet)
        self.mapping_ = dict(zip(alphabet, range(self.mapping_size_)))

    def transform(self, X: str) -> List[int]:
        features = [0] * self.mapping_size_

        for char in X:
            if char in self.mapping_:
                features[self.mapping_[char]] += 1

        return features
