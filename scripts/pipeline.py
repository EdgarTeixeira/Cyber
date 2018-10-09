import re
import tarfile
import unicodedata
from typing import List


def data_archive(archive: str, encoding: str) -> str:
    corpus: List[str] = []

    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            fhandler = tar.extractfile(item)
            if fhandler is not None:
                content = fhandler.read()\
                                  .decode(encoding)\
                                  .replace("\\x93", '"')\
                                  .replace("\\x94", '"')\
                                  .replace("\\x97", '-')
                corpus.append(content)
                fhandler.close()

    return '\n\n'.join(corpus)


def remove_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != 'Mn')


def remove_punctuations(text: str) -> str:
    # TODO: Should ignore numbers? Like 3.0 or 3,0
    return re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\\\]^_`{|}~]+", " ", text)


def remove_case(text: str) -> str:
    return text.lower()


def flatten_text(text: str) -> str:
    return re.sub("\\s+", " ", text)


def build_ngrams(text: str, size: int) -> List[str]:
    if size <= 0:
        raise ValueError("size must be greater than 0")
    return [text[idx:idx + size] for idx in range(len(text) - size + 1)]
