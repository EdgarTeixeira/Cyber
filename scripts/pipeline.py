import random
import re
import tarfile
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


def remove_punctuations(text: str) -> str:
    # TODO: Should ignore numbers? Like 3.0 or 3,0
    return re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\\\]^_`{|}~“”—‘’']+", " ", text)


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
