"""Download a fixed QQP training subset for repeatable, short lab experiments."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'task_datasets' / 'data' / 'lab'
QQP_URL = 'https://dl.fbaipublicfiles.com/glue/data/QQP.zip'
VOCAB_URL = 'https://huggingface.co/bert-large-cased/resolve/main/vocab.txt'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, path):
    path = Path(path)
    if path.is_file():
        return
    temporary = path.with_suffix(path.suffix + '.part')
    print('Downloading', url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
        shutil.copyfileobj(response, out)
    temporary.replace(path)


def prepare(rows=4096):
    if rows < 8:
        raise ValueError('need at least 8 training rows')
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'QQP.zip'
    download(QQP_URL, archive)
    vocab = DATA / 'vocab.txt'
    download(VOCAB_URL, vocab)
    tokens = vocab.read_text(encoding='utf-8').splitlines()
    if not all(token in tokens for token in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')):
        raise ValueError('downloaded vocabulary is not a BERT vocabulary')
    output = DATA / 'train.tsv'
    selected = []
    counts = [0, 0]
    with zipfile.ZipFile(archive) as package:
        with io.TextIOWrapper(package.open('QQP/train.tsv'), encoding='utf-8') as source:
            header = next(source).rstrip('\r\n')
            if len(header.split('\t')) != 6:
                raise ValueError('unexpected QQP header')
            for line in source:
                fields = line.rstrip('\r\n').split('\t')
                if (len(fields) != 6 or not fields[0].isdigit() or not fields[3].strip()
                        or not fields[4].strip() or fields[5] not in ('0', '1')):
                    continue
                selected.append('\t'.join(fields))
                counts[int(fields[5])] += 1
                if len(selected) == rows:
                    break
    if len(selected) != rows or not all(counts):
        raise ValueError('not enough valid QQP rows or a class is missing')
    output.write_text(header + '\n' + '\n'.join(selected) + '\n', encoding='utf-8', newline='\n')
    manifest = dict(dataset='QQP', selection='first valid training rows; systems benchmark subset',
                    rows=rows, class_counts=counts, source=QQP_URL,
                    train_sha256=digest(output), vocab_sha256=digest(vocab))
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('QQP ready:', output, '| rows:', rows, '| class counts:', counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=4096)
    prepare(parser.parse_args().rows)
