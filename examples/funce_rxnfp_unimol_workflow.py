"""Funce end to end from a reaction SMILES, in one Step.

The DataFrame is the sequence-embedding database; 
the reaction is a constructor argument. Everything between -- encoding the reaction, broadcasting it across every
row, scoring -- lives in the step, in
enzymetk/predict_funce_step_rxnfp_unimol_workflow.py.

The protein side is not computed here: the databases already carry esm3_mean (1536-d,
ESM3-open). Building that column is examples/esm3.py's job.

    Local:   python examples/funce_rxnfp_unimol_workflow.py
    Docker:  docker build -f examples/docker/Dockerfile.funce_rxnfp_unimol_workflow -t enzymetk-funce-rxnfp-unimol-workflow .
             docker run -v "$(pwd)/examples/data:/app/examples/data" enzymetk-funce-rxnfp-unimol-workflow
"""
import os
# Set before numpy/MKL loads, as every other example in examples/ does.
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import logging
from pathlib import Path

import pandas as pd

from enzymetk import Funce_rxnfp_unimol

# The step reports what it is doing through logging -- which molecules it embedded,
# whether the checkpoint was cached, when a side gets summed.
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
# numba (via unimol_tools) emits its bytecode analysis at DEBUG, and something in that
# import chain drops the root logger to DEBUG -- burying the output under ~5000 lines.
logging.getLogger('numba').setLevel(logging.WARNING)

# Serine + propiophenone -> the aldol adduct. Two substrate molecules: this is the
# reaction that exercises the split-and-sum.
REACTION = 'N[C@H](C(O)=O)CO.CCC(C1=CC=CC=C1)=O>>CC(C(C2=CC=CC=C2)=O)C[C@@H](C(O)=O)N'

HERE = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get('DATA_DIR', HERE / 'data'))
MODEL_DIR = DATA_DIR / 'funce_models'            # the 4 EC-level checkpoints (1.5 GB)
WEIGHT_DIR = DATA_DIR / 'unimol_weights'         # UniMol checkpoint cache (659 MB)
OUT_DIR = DATA_DIR / 'output'

SEQ_DBS = [
    DATA_DIR / 'sequence_embeddings' / 'enzymes_sample_10_but_9.pkl',   # 9 rows
    DATA_DIR / 'sequence_embeddings' / 'enzyme_embedding.pkl',          # 5 rows
]


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loading is the caller's job. `database` is the column enzymetk's own multi-database
    # step uses (similarity_sequence_and_structure_step.py), so results stay traceable.
    db = pd.concat([pd.read_pickle(p).assign(database=p.name) for p in SEQ_DBS],
                   ignore_index=True)
    print(f'loaded {len(db)} sequences from {len(SEQ_DBS)} databases')

    # run the step, which encodes the reaction, broadcasts it across every row, and scores
    scored = db << Funce_rxnfp_unimol(
        REACTION,
        model_dir=str(MODEL_DIR),
        unimol_weights_dir=str(WEIGHT_DIR),
        download_if_missing=True,     # the library default is False; this example opts in
    )
    ranked = scored.sort_values('Funce_prediction', ascending=False).reset_index(drop=True)

    out_path = OUT_DIR / 'funce_rxnfp_unimol_workflow_results.csv'
    ranked.to_csv(out_path, index=False)

    print(f'\nreaction: {REACTION}')
    print(f'\n{len(ranked)} sequences ranked by predicted activity:\n')
    print(ranked[['Entry', 'database', 'Funce_prediction', 'Funce_std_preds']]
          .to_string(index=False))
    print(f'\nwrote {out_path}')
