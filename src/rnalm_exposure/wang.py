"""Load Wang et al. (2026)'s structure sets (Zenodo 18218776) from their zip."""
import io
import pickle
import zipfile

import numpy as np

# Wang's files are Python pickles. This unpickler rebuilds plain lists, strings and numpy arrays and
# refuses everything else, so the file can't run code when it is loaded.
ALLOWED = {("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"),
           ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def load_set(zip_path, member, name):
    with zipfile.ZipFile(zip_path) as z:
        records = SafeUnpickler(io.BytesIO(z.read(member))).load()
    return [{"seq_id": f"{name}:{r[0]}", "sequence": r[1], "contacts": np.asarray(r[2]), "keep": list(r[3])}
            for r in records]
