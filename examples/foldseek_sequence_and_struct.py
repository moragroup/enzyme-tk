"""
Example: FoldSeek similarity search — sequence and structure modes.

Runs two modes using a single query protein (lysozyme, PDB 1AKI):
  1. Sequence mode  — search via ProstT5 (no structure files needed)
  2. Structure mode  — search with a CIF/PDB file

Both modes accept any combination of databases:
  - FoldSeekDatabase enum values (PDB, Alphafold/Swiss-Prot, CUSTOM)
  - Named pre-built custom databases as plain strings (e.g. "custom_v2")

Pre-built custom databases are resolved under database_root_path as:
  root/{name}/{name}   (e.g. data/foldseek_db/custom_v2/custom_v2.index)

Docker:  docker run -v $(pwd)/examples/data:/data enzymetk-foldseek
Local:   DATA_DIR=examples/data python examples/foldseek_seq_struct.py
"""

import os
import urllib.request
from pathlib import Path

import pandas as pd

from enzymetk.similarity_sequence_and_structure_step import FoldSeek, FoldSeekDatabase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
WEIGHTS_DIR = DATA_DIR / "foldseek_models" / "weights"
DB_ROOT = DATA_DIR / "foldseek_db"
STRUCTURES_DIR = DATA_DIR / "structures"
OUTPUT_DIR = DATA_DIR / "output"

for directory in (STRUCTURES_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference sequences (first entry is the default query)
# ---------------------------------------------------------------------------
REFERENCE_SEQUENCES = [
    {
        "id": "1AKI",
        "sequence": (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINS"
        "RWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQ"
        "AWIRGCRL"
    )
    },
    {
        "id": "1LYZ", 
        "sequence": (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINS"
        "RWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQ"
        "AWIRGCRL"
    )
    },
    {
        "id": "2VB1", 
        "sequence": (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINS"
        "RWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGDGMNAWVAWRNRCKGTDVQ"
        "AWIRGCRL"
    )
    },
    {
        "id": "A0A009IHW8", 
        "sequence": (
        "MSLEQKKGADIISKILQIQNSIGKTTSPSTLKTKLSEISRKEQENARIQSKLSDLQKKK"
        "IDIDNKLLKEKQNLIKEEILERKKLEVLTKKQQKDEIEHQKKLKREIDAIKASTQYITD"
        "VSISSYNNTIPETEPEYDLFISHASEDKEDFVRPLAETLQQLGVNVWYDEFTLKVGDSL"
        "RQKIDSGLRNSKYGTVVLSTDFIKKDWTNYELDGLVAREMNGHKMILPIWHKITKNDVL"
        "DYSPNLADKVALNTSVNSIEEIAHQLADVILNR"
    )
    },
    {
        "id": "A0A067CMC7", 
        "sequence": (
        "MLEVPVWIPILAFAVGLGLGLLIPHLQKPFQRFSTVNDIPKEFFEHERTLRGKVVS"
        "VTDGDTIRVRHVPWLANGDGDFKGKLTETTLQLRVAGVDCPETAKFGRTGQPFGEE"
        "AKAWLKGELQDQVVSFKLLMKDQYSRAVCLVYYGSWAAPMNVSEELLRHGYANIYR"
        "QSGAVYGGLLETFEALEAEAREKRVNIWSLDKRETPAQYKARK"
    )
    },
]



def ensure_cif(pdb_id: str) -> str:
    """Download a CIF from RCSB if not already cached."""
    cif_path = STRUCTURES_DIR / f"{pdb_id}.cif"
    if not cif_path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        print(f"Downloading {pdb_id}.cif ...")
        urllib.request.urlretrieve(url, str(cif_path))
    return str(cif_path)


def print_results(result_df: pd.DataFrame, label: str):
    """Print a summary grouped by database and save to CSV."""
    print(f"\n{'='*60}")
    print(f"  {label}: {len(result_df)} hits across {result_df['database'].nunique()} database(s)")
    print(f"{'='*60}")
    for db_name, group in result_df.groupby("database"):
        print(f"\n  {db_name}: {len(group)} hits")
        print(group.head(5).to_string(index=False))

    output_filename = label.lower().replace(" ", "_") + ".csv"
    result_df.to_csv(OUTPUT_DIR / output_filename, index=False)


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------
def run_sequence_search(queries: list[dict]) -> pd.DataFrame:
    """Sequence-only search against the specified databases."""
    query_ids = {q["id"] for q in queries}
    references = [s for s in REFERENCE_SEQUENCES if s["id"] not in query_ids]
    df = pd.DataFrame([
        *[{"Entry": q["id"], "Sequence": q["sequence"], "label": "query"}
          for q in queries],
        *[{"Entry": s["id"], "Sequence": s["sequence"], "label": "reference"}
          for s in references],
    ])
    step = FoldSeek(
        id_column_name="Entry",
        sequence_column_name="Sequence",
        prostt5_weights_path=str(WEIGHTS_DIR),

        #databases=[FoldSeekDatabase.PDB, FoldSeekDatabase.AFDB_SWISSPROT, FoldSeekDatabase.CUSTOM],
        # if you have a pre-built custom database under 
        # DB_ROOT (e.g. data/foldseek_db/custom_v2/), 
        # you can specify it by name as a plain string instead of using the CUSTOM enum
        databases=[FoldSeekDatabase.PDB, FoldSeekDatabase.AFDB_SWISSPROT, FoldSeekDatabase.CUSTOM, "CUSTOM2"],

        database_root_path=str(DB_ROOT),
        label_column_name="label",
    )

    result_df = step.execute(df)
    label = f"sequence_search_{'_'.join(q['id'] for q in queries)}"
    print_results(result_df, label)
    return result_df


def run_structure_search(query: dict) -> pd.DataFrame:
    """Structure search (CIF file) against the specified databases."""
    query_id = query["id"]
    cif_path = ensure_cif(query_id)
    references = [s for s in REFERENCE_SEQUENCES if s["id"] != query_id]
    df = pd.DataFrame([
            {"Entry": query_id,
            "Sequence": query["sequence"],
            "structure": cif_path,
            "label": "query"
            },
        *[{"Entry": s["id"], "Sequence": s["sequence"], "structure": "", "label": "reference"}
          for s in references],
    ])
    step = FoldSeek(
        id_column_name="Entry",
        sequence_column_name="Sequence",
        prostt5_weights_path=str(WEIGHTS_DIR),
        databases=[FoldSeekDatabase.PDB, FoldSeekDatabase.AFDB_SWISSPROT, FoldSeekDatabase.CUSTOM],
        database_root_path=str(DB_ROOT),
        structure_column_name="structure",
        label_column_name="label",
    )
    result_df = step.execute(df)
    label = f"structure_search_{query_id}"
    print_results(result_df, label)
    return result_df


if __name__ == "__main__":
    
    #run_sequence_search([REFERENCE_SEQUENCES[0]]) # 1AKI
    
    #run_structure_search(REFERENCE_SEQUENCES[0]) # 1AKI
    
    # run sequence search with multiple queries at once
    run_sequence_search([REFERENCE_SEQUENCES[3], REFERENCE_SEQUENCES[4]])  # A0A009IHW8 + A0A067CMC7
    
   
    print(f"\n{'='*60}\n  DONE — results saved to {OUTPUT_DIR}\n{'='*60}")
