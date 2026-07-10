"""Funce ensemble activity prediction example.

Funce is *prediction-only*: the input DataFrame must already contain the
embedding columns produced by the encoding steps:

    esm3_mean              protein embedding   (EmbedESM3)
    rxnfp                  reaction fingerprint (RxnFP)
    substrate_unimol_repr  substrate embedding  (UniMol)
    product_unimol_repr    product embedding    (UniMol)

A single reaction is scored against many proteins by broadcasting the reaction's
rxnfp / substrate / product embeddings across every protein row *before* calling
Funce (as the original notebook/script do). ``examples/Funce_pairs.pkl`` is a
tiny pre-encoded, already-broadcast DataFrame (5 proteins vs. the DEHP->MEHP
reaction) for a quick smoke test.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from pathlib import Path
import pandas as pd
from enzymetk import Funce, Save

# Resolve data paths relative to this script so it runs from any directory.
HERE = Path(__file__).parent.resolve()

df = pd.read_pickle(HERE / 'Funce_pairs.pkl')  # esm3_mean + rxnfp + *_unimol_repr

df = df << (Funce('Entry', name='DEHP-MEHP') >> Save(str(HERE / 'Funce_out.pkl')))

print(df[['Entry', 'DEHP-MEHP_prediction', 'DEHP-MEHP_std_preds']])
