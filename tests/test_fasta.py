import gzip

import pytest

from rnalm_exposure.fasta import count_unique_sequences, iter_gz_fasta, normalize


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "x.fasta.gz"
    # a and c are the same sequence; b differs from them only in case and T/U; d is distinct.
    with gzip.open(path, "wb") as f:
        f.write(b">a desc\nACGT\nAC\n>b\nacgu\nac\n>c\nACGTAC\n>d\nGG\n")
    return path


def test_iter_joins_wrapped_lines(fasta):
    records = list(iter_gz_fasta(fasta))
    assert records[0] == (b"a desc", b"ACGTAC")
    assert len(records) == 4


def test_normalize():
    assert normalize(b"acgT") == b"ACGU"


def test_count_unique_sequences(fasta):
    assert count_unique_sequences(fasta) == {"records": 4, "unique_raw": 3, "unique_normalized": 2}


def test_truncated_file_raises(tmp_path, fasta):
    bad = tmp_path / "bad.fasta.gz"
    bad.write_bytes(fasta.read_bytes()[:-10])
    with pytest.raises(RuntimeError):
        count_unique_sequences(bad)
