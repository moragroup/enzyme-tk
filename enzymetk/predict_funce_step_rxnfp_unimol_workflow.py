"""Score a sequence-embedding database against one reaction, in one Step.

The DataFrame is the database; the reaction is a constructor argument::

    from enzymetk import Funce_rxnfp_unimol

    db = pd.read_pickle('enzyme_embedding.pkl')     # Entry, Sequence, esm3_mean
    scored = db << Funce_rxnfp_unimol(rxn_smiles, model_dir=..., unimol_weights_dir=...)

This is a recipe pinned to one encoder stack, not a general-purpose step. The four
widths are fixed by the trained checkpoints -- read off the layer shapes in
``funce_models/*_checkpoint.pth``, all four models agreeing::

    forward_enzyme     nn.Linear(in=1536, out=1024)    esm3_mean   <- must already be in df
    forward_reaction   nn.Linear(in=256,  out=1024)    rxnfp       <- RxnFP, added here
    forward_substrate  nn.Linear(in=768,  out=1024)    substrate_unimol_repr
    forward_product    nn.Linear(in=768,  out=1024)    product_unimol_repr

``esm3_mean`` is an INPUT, not something this step produces: build it once with
``EmbedESM3`` (see examples/esm3.py) and keep it in the database. ESM3-open is the only
protein embedder that fits -- no ESM-2 model emits 1536 (they are 320 / 480 / 640 /
1280 / 2560 / 5120), so watch for ``EmbedESM`` defaulting to esm2_t36_3B_UR50D.

rxnfp, not DRFP: rxnfp emits dense continuous 256-d that drops straight into
forward_reaction, while DRFP folded to 256 is *accepted* and silently reranks. DRFP is
not reachable from here.

Multi-molecule sides are SUMMED. ``A.B.C`` is split on the dot, each molecule embedded
on its own, and the vectors reduced to one by :func:`combine_molecule_embeddings`. Some
reduction is unavoidable -- there is exactly one 768-d slot per side -- but which one is
a choice, and nothing in this repo records what the training pipeline used, so a warning
is logged whenever it bites.

Two caveats worth knowing. ``execute`` mutates the frame you pass it (see its
docstring). And Funce's cross-attention softmaxes over the batch, so a row's score
depends on which other rows went through the same call -- screening a concatenated
database is not the same as screening each database separately and merging.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from enzymetk.step import Step
from enzymetk.embedchem_rxnfp_step import RxnFP
from enzymetk.embedchem_unimol_step import UniMol
from enzymetk.predict_Funce_step import Funce, ENZYME_COLS, REACTION_COLS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Widths the trained weights were fit on.
EXPECTED_WIDTHS = {
    'esm3_mean': 1536,
    'rxnfp': 256,
    'substrate_unimol_repr': 768,
    'product_unimol_repr': 768,
}

# Not configurable: unimolv2/164m is what emits 768, and UniMol's weights_dir= only
# supports unimolv2 anyway.
UNIMOL_MODEL = 'unimolv2'
UNIMOL_SIZE = '164m'

# Columns Funce leaks onto the frame. retransform_scaled_predictions() mutates df in
# place once per ensemble member, so what survives is the LAST model's values -- not an
# average -- sitting next to the correct {name}_*_mean columns. Named exactly rather
# than prefix-matched, so a caller's own 'pred_*' column is never collateral damage.
LEAKED_COLS = (
    {f'pred_{c}' for c in ENZYME_COLS + REACTION_COLS}
    | {f'inverse_transformed_pred_{c}' for c in ENZYME_COLS + REACTION_COLS}
    | {'pred_Activity'}
)


# --- how a multi-molecule side becomes one vector ----------------------------
#
# 'A.B.C' is embedded one molecule at a time and reduced to the single 768-d vector
# forward_substrate / forward_product were fit on. Handing UniMol the dot-joined string
# instead embeds it as ONE disconnected molecule -- a confident, meaningless result.
#
# The reduction is a SUM and it is the only thing this function does, so changing it is
# a one-line edit. Deliberately not a constructor option: it changes what every score
# means, so it should be a visible edit here rather than a knob two callers set
# differently. Drop-in alternatives:
#
#     mean   return stacked.mean(axis=0).astype(np.float32)   # scale-preserving
#     max    return stacked.max(axis=0).astype(np.float32)
#
# sum([v]) == v, so single-molecule sides are byte-for-byte unchanged.
def combine_molecule_embeddings(vectors):
    """Reduce one side's per-molecule UniMol embeddings to a single (768,) vector."""
    stacked = np.stack([np.asarray(v).flatten().astype(np.float32) for v in vectors])
    return stacked.sum(axis=0).astype(np.float32)   # <-- the reduction


def split_reaction(reaction):
    """``'A.B>>C'`` -> ``(['A', 'B'], ['C'])``.

    Splits on '>' and requires three blocks, not on '>>': reaction SMILES are
    reactants>agents>products, so ``'A>B>C'.split('>>')`` returns the WHOLE string as
    both substrate and product.
    """
    parts = str(reaction).strip().split('>')
    if len(parts) != 3:
        raise ValueError(f'{reaction!r} is not a reaction SMILES: expected '
                         f"reactants>agents>products (e.g. 'A.B>>C'), got "
                         f"{len(parts)} '>'-separated block(s).")
    if parts[1].strip():
        logger.warning('Agent block %r is ignored: Funce scores substrate and product '
                       'only.', parts[1].strip())
    sides = []
    for side, which in ((parts[0], 'substrate'), (parts[2], 'product')):
        molecules = [m.strip() for m in side.split('.') if m.strip()]
        if not molecules:
            raise ValueError(f'the {which} side of {reaction!r} is empty.')
        sides.append(molecules)
    return sides[0], sides[1]


def check_funce_models(model_dir):
    """Fail before the 659 MB UniMol load if the Funce checkpoints are absent."""
    model_dir = Path(model_dir)
    names = [f'run_easy_0-50_ESRP_{lvl}_model_1_500000_{suffix}'
             for lvl in range(1, 5) for suffix in ('checkpoint.pth', 'conf.pkl')]
    missing = [n for n in names if not (model_dir / n).is_file()]
    if len(missing) == len(names):
        raise FileNotFoundError(
            f'No Funce checkpoints in {model_dir} (they are not downloadable). '
            f'Expected 8 files:\n  ' + '\n  '.join(names))
    if missing:
        # Funce tolerates a partial ensemble but the predictions will not match.
        logger.warning('%d of 8 Funce files missing in %s; the ensemble will be smaller '
                       'and predictions will differ: %s',
                       len(missing), model_dir, ', '.join(missing))


def resolve_unimol_weights(weights_dir, download_if_missing=False):
    """Return a weights dir holding the UniMol checkpoint, or None to let unimol_tools decide."""
    try:
        from unimol_tools.config import MODEL_CONFIG_V2      # needs unimol_tools >= 0.1.3
    except ImportError as e:
        raise ImportError('Funce_rxnfp_unimol needs unimol-tools '
                          '(pip install unimol_tools).') from e

    if weights_dir is None:
        if not download_if_missing:
            raise ValueError(
                'unimol_weights_dir is required unless download_if_missing=True: with no '
                'directory to check, unimol_tools downloads ~659 MB into its own '
                'site-packages folder.')
        return None

    relative_path = MODEL_CONFIG_V2['weight'][UNIMOL_SIZE]   # 'modelzoo/164M/checkpoint.pt'
    checkpoint = Path(weights_dir) / relative_path
    os.environ['UNIMOL_WEIGHT_DIR'] = str(weights_dir)       # the only way to relocate the cache
    if checkpoint.is_file():
        logger.info('UniMol checkpoint: %s (cached)', checkpoint)
        return str(weights_dir)
    if not download_if_missing:
        raise FileNotFoundError(
            f'UniMol checkpoint not found at {checkpoint}. Place {relative_path!r} under '
            f'{weights_dir}, or pass download_if_missing=True to fetch it (~659 MB).')

    from unimol_tools.weights import weight_download_v2
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    logger.info('UniMol checkpoint missing; downloading %s -> %s', relative_path, weights_dir)
    try:
        weight_download_v2(relative_path, str(weights_dir))
    except Exception as e:
        raise RuntimeError(
            f'Downloading the UniMol checkpoint to {weights_dir} failed: '
            f'{type(e).__name__}: {e}. Check network access to huggingface.co and ~659 MB '
            'of free space.') from e
    return str(weights_dir)


def validate_embeddings(df, protein_emb_col='esm3_mean'):
    """Check all four embedding columns against the widths the models were fit on.

    Runs before Funce, which reads these by name and validates none of them -- a wrong
    width otherwise dies inside the first matmul as "mat1 and mat2 shapes cannot be
    multiplied", naming neither the column nor where it came from.
    """
    for col, width in EXPECTED_WIDTHS.items():
        name = protein_emb_col if col == 'esm3_mean' else col
        if name not in df.columns:
            raise ValueError(f'{name!r} missing; Funce needs {sorted(EXPECTED_WIDTHS)}.')
        value = df[name].iloc[0]
        if value is None:
            raise ValueError(f'{name!r} is None in the first row.')
        found = np.asarray(value).flatten().shape[0]
        if found != width:
            hint = ('' if col != 'esm3_mean' else
                    f' {name} must come from ESM3-open (1536-d) -- ESM-2 gives '
                    '320/480/640/1280/2560/5120 and is not a substitute.')
            raise ValueError(f'{name!r} is {found}-d but Funce was trained on '
                             f'{width}-d.{hint}')


def _encode_reaction(reaction, weights_dir=None):
    """Reaction SMILES -> the three vectors Funce reads, keyed by column name."""
    substrates, products = split_reaction(reaction)
    for which, molecules in (('substrate', substrates), ('product', products)):
        logger.info('%s: %s', which, ' + '.join(molecules))
        if len(molecules) > 1:
            logger.warning(
                '%s side has %d molecules; their UniMol embeddings are SUMMED into one '
                '768-d vector (see combine_molecule_embeddings). Funce has a single '
                '768-d slot per side, so some reduction is required; nothing records '
                'which one the training pipeline used.', which, len(molecules))

    # RxnFP shells out with cmd[0] == 'python', not sys.executable, so make a bare
    # `python` resolve to this interpreter. No-op where it already does.
    bindir = str(Path(sys.executable).parent)
    if bindir not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    try:
        # A bare one-column frame: RxnFP round-trips through to_csv/read_csv, so it must
        # never see an array column. env_name=None skips `conda run`; tmp_dir must be a
        # real path, because left None the step f-strings the TemporaryDirectory object
        # into the filename.
        with TemporaryDirectory() as tmp:
            rxn_df = pd.DataFrame({'reaction': [reaction]}) << RxnFP(
                'reaction', 1, env_name=None, tmp_dir=tmp)
    except subprocess.CalledProcessError as e:
        # Step.run() uses check=True, which raises before its own logging runs -- so the
        # child's stderr, the only thing saying why rxnfp failed, is otherwise lost.
        raise RuntimeError(
            f'RxnFP failed (exit {e.returncode}). Its stderr:\n'
            f'{(e.stderr or "").strip() or "(none)"}') from e
    except Exception as e:
        raise RuntimeError(f'RxnFP failed to encode the reaction: '
                           f'{type(e).__name__}: {e}') from e

    # One UniMol call for every molecule: it rebuilds UniMolRepr (reloading the 659 MB
    # checkpoint) on every execute(), and always writes to the fixed 'unimol_repr'.
    unique = list(dict.fromkeys(substrates + products))
    try:
        mol_df = pd.DataFrame({'smiles': unique}) << UniMol(
            'smiles', unimol_model=UNIMOL_MODEL, unimol_size=UNIMOL_SIZE,
            weights_dir=weights_dir)
    except (ValueError, FileNotFoundError, ImportError):
        raise                       # UniMol's own messages are already specific
    except Exception as e:
        raise RuntimeError(
            f'UniMol failed embedding {len(unique)} molecule(s): {type(e).__name__}: {e}. '
            'A truncated checkpoint is the usual cause.') from e

    embeddings = {}
    for smiles, value in zip(mol_df['smiles'], mol_df['unimol_repr']):
        if value is None:
            # UniMol logs and returns None rather than raising. Stop here: a missing
            # molecule would drop out of the sum and the score would still look fine.
            raise ValueError(f'UniMol could not embed {smiles!r} (see the log above).')
        embeddings[smiles] = np.asarray(value).flatten().astype(np.float32)   # cls_repr is nested

    # Lists, not sets: 'A.A' must count twice even though UniMol embedded A once.
    return {
        'rxnfp': np.asarray(rxn_df['rxnfp'].iloc[0]).flatten().astype(np.float32),
        'substrate_unimol_repr': combine_molecule_embeddings([embeddings[m] for m in substrates]),
        'product_unimol_repr': combine_molecule_embeddings([embeddings[m] for m in products]),
    }


class Funce_rxnfp_unimol(Step):
    """Rank a sequence-embedding database by predicted activity on one reaction."""

    def __init__(self, reaction: str,
                 model_dir,
                 unimol_weights_dir=None,
                 download_if_missing: bool = False,
                 id_col: str = 'Entry',
                 protein_emb_col: str = 'esm3_mean'):
        super().__init__()
        self.reaction = reaction
        # Required: Funce's own default points at repo/data/Funce/models, which does not
        # exist, so defaulting only ever yields a confusing "no models loaded".
        self.model_dir = Path(model_dir)
        self.unimol_weights_dir = unimol_weights_dir
        self.download_if_missing = download_if_missing
        self.id_col = id_col
        self.protein_emb_col = protein_emb_col

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """df IS the sequence-embedding database: id_col, protein_emb_col, and the rest.

        Mutates df in place -- it gains rxnfp / substrate_unimol_repr /
        product_unimol_repr, then Funce's own output columns. No defensive copy: Funce
        mutates in place regardless. Said out loud because those three are ndarray
        columns, and they have to be dropped before any json.dumps of the result.
        """
        for col in (self.id_col, self.protein_emb_col):
            if col not in df.columns:
                raise ValueError(f'{col!r} missing from the database '
                                 f'(has {list(df.columns)}).')
        if df.empty:
            raise ValueError('the database has no rows.')

        # Fail fast, cheapest first: neither of these loads a model.
        check_funce_models(self.model_dir)
        weights_dir = resolve_unimol_weights(self.unimol_weights_dir,
                                             self.download_if_missing)

        for col, vector in _encode_reaction(self.reaction, weights_dir).items():
            df[col] = [vector] * len(df)
        validate_embeddings(df, self.protein_emb_col)

        logger.info('Funce: scoring %d sequence(s)', len(df))
        try:
            scored = df << Funce(self.id_col, protein_emb_col=self.protein_emb_col,
                                 model_dir=str(self.model_dir))
        except Exception as e:
            raise RuntimeError(
                f'Funce failed scoring {len(df)} sequence(s): {type(e).__name__}: {e}. '
                f'Checkpoints: {self.model_dir} (torch.load unpickles model objects, so '
                'enzymetk.Funce_model must be importable). The four widths were '
                'validated before this call.') from e

        return scored.drop(columns=[c for c in scored.columns if c in LEAKED_COLS])
