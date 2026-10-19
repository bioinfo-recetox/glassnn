"""Write ``protein_localization.fasta``, the data of Colab notebook C.

Usage: ``uv run python examples/_fetch_proteins.py`` (needs internet access).
The file in the repository was written with UniProt release 2026_03; a later
release gives a slightly different file, which is why the file is shipped
instead of being queried by the notebook (``PLAN.md``, decision 51).

The data are reviewed human proteins (UniProtKB/Swiss-Prot) with an
experimentally supported subcellular location, either in the nucleus and not
in the cytoplasm, or the other way round, of length 50 to 1000. From each
class, ``PER_CLASS`` proteins are drawn at random with a fixed seed. Each
FASTA header is ``>ACCESSION LABEL``, with LABEL ``nucleus`` or
``cytoplasm``.

UniProt data are distributed under the CC BY 4.0 license
(https://www.uniprot.org/help/license).
"""

import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "protein_localization.fasta"
PER_CLASS = 250
SEED = 0

# UniProt subcellular-location vocabulary: SL-0191 nucleus, SL-0086 cytoplasm.
BASE = "organism_id:9606 AND reviewed:true AND length:[50 TO 1000]"
QUERIES = {
    "nucleus": f"{BASE} AND cc_scl_term_exp:SL-0191 NOT cc_scl_term_exp:SL-0086",
    "cytoplasm": f"{BASE} AND cc_scl_term_exp:SL-0086 NOT cc_scl_term_exp:SL-0191",
}


def fetch(query: str) -> tuple[dict[str, str], str]:
    """All matching sequences as {accession: sequence}, and the UniProt release."""
    url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(
        {"query": query, "format": "tsv", "fields": "accession,sequence"}
    )
    with urllib.request.urlopen(url) as response:
        release = response.headers["X-UniProt-Release"]
        lines = response.read().decode().splitlines()[1:]  # skip the header
    return dict(line.split("\t") for line in lines), release


def main() -> None:
    rng = np.random.default_rng(SEED)
    records = []
    for label, query in QUERIES.items():
        sequences, release = fetch(query)
        accessions = sorted(sequences)  # the stream order is not guaranteed
        chosen = rng.choice(accessions, size=PER_CLASS, replace=False)
        records += [(accession, label, sequences[accession]) for accession in chosen]
        print(f"{label}: {len(accessions)} proteins, {PER_CLASS} drawn ({release})")
    records.sort()
    OUTPUT.write_text("".join(f">{a} {label}\n{s}\n" for a, label, s in records))
    print(f"wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
