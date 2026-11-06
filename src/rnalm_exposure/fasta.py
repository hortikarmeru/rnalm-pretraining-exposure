"""FASTA helpers for multi-GB gzipped sequence databases."""
import hashlib
import subprocess


def iter_gz_fasta(path):
    """Yield (header, sequence) byte strings from a gzipped FASTA.

    gzip -dc is several times faster than Python's gzip module on multi-GB files.
    """
    proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)
    header, chunks = None, []
    for line in proc.stdout:
        if line.startswith(b">"):
            if header is not None:
                yield header, b"".join(chunks)
            header, chunks = line[1:].rstrip(), []
        else:
            chunks.append(line.rstrip())
    if header is not None:
        yield header, b"".join(chunks)
    if proc.wait() != 0:
        raise RuntimeError(f"gzip failed on {path} (incomplete download?)")


def normalize(seq):
    """RNA-FM's preprocessing: uppercase, T -> U."""
    return seq.upper().replace(b"T", b"U")


def count_unique_sequences(path):
    """Count records, distinct sequences as written, and distinct sequences after normalize().

    Sequences are stored as 16-byte MD5 digests so ~36M of them fit in a few GB.
    """
    raw_seen, norm_seen = set(), set()
    records = 0
    for _, seq in iter_gz_fasta(path):
        records += 1
        raw_seen.add(hashlib.md5(seq).digest())
        norm_seen.add(hashlib.md5(normalize(seq)).digest())
    return {"records": records, "unique_raw": len(raw_seen), "unique_normalized": len(norm_seen)}
