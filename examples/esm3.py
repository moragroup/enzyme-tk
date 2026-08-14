"""Embed protein sequences with ESM3-open -> the 1536-d column Funce consumes.

Sequences in -> DataFrame with an `esm3_mean` column of (1536,) float32 out.

ESM3 is not one option among several here. Funce's forward_enzyme is
nn.Linear(in=1536) and no ESM-2 model emits 1536 (they are 320 / 480 / 640 /
1280 / 2560 / 5120), so ESM3-open is the only protein embedder that feeds it --
see the table in examples/encode_reaction.py. That is why the column in
examples/Funce_pairs.pkl is called esm3_mean.

Pair this with examples/encode_reaction.py (the reaction side) to build a
complete Funce input from sequences and a reaction SMILES, rather than from the
pre-computed pickle.

Weights: ~5.4 GB, downloaded once on first use and cached under HF_HOME. See
ensure_esm3_weights() below.

Runs on CPU. Set ESM3_DEVICE=cuda (or pass device=) on a GPU host.
"""
import os

# Both must be set before enzymetk is imported. MKL_THREADING_LAYER matches the
# other examples; HF_HOME has to beat the import because huggingface_hub freezes
# its cache paths into module constants at import time, and enzymetk/__init__
# pulls huggingface_hub in transitively. Setting it inside a function is too late.
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from pathlib import Path

HERE = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get('DATA_DIR', HERE / 'data'))
os.environ.setdefault('HF_HOME', str(DATA_DIR / 'hf'))

import numpy as np
import pandas as pd

from enzymetk.embedprotein_esm3_step import EmbedESM3
from enzymetk.save_step import Save


ESM3_REPO = 'EvolutionaryScale/esm3-sm-open-v1'

# No partial download here, and it is not for lack of trying. Only 2.9 GB of the
# 5.4 GB is on this code path: the structure and function decoders (1.2 GB each)
# are built lazily by get_structure_decoder() / get_function_decoder(), which only
# decode() reaches, and this step encodes and forward-samples but never decodes.
# Fetching just the needed files with allow_patterns does work -- and then buys
# nothing, because ESM3.from_pretrained -> data_root() runs its own
# snapshot_download(repo_id=...) with no patterns and pulls the rest anyway.
# Measured: a filtered 18-file fetch still ends at 5.1 GB on disk with all four
# .pth files present. Short of setting HF_HUB_OFFLINE and hoping, the full
# snapshot is the honest cost.

OUTPUT_DIR = DATA_DIR / 'output'
OUTPUT_PKL = OUTPUT_DIR / 'esm3_embeddings.pkl'

id_col = 'Entry'
seq_col = 'Sequence'
label_col = 'ActiveSite'
rows = [['AXE2_TALPU', '10', 'MHSKFFAASLLGLGAAAIPLEGVMEKRSCPAIHVFGARETTASPGYGSSSTVVNGVLSAYPGSTAEAINYPACGGQSSCGGASYSSSVAQGIAAVASAVNSFNSQCPSTKIVLVGYSQGGEIMDVALCGGGDPNQGYTNTAVQLSSSAVNMVKAAIFMGDPMFRAGLSYEVGTCAAGGFDQRPAGFSCPSAAKIKSYCDASDPYCCNGSNAATHQGYGSEYGSQALAFVKSKLG'],
        ['AXE2_GEOSE', '1|2', 'MKIGSGEKLLFIGDSITDCGRARPEGEGSFGALGTGYVAYVVGLLQAVYPELGIRVVNKGISGNTVRDLKARWEEDVIAQKPDWVSIMIGINDVWRQYDLPFMKEKHVYLDEYEATLRSLVLETKPLVKGIILMTPFYIEGNEQDPMRRTMDQYGRVVKQIAEETNSLFVDTQAAFNEVLKTLYPAALAWDRVHPSVAGHMILARAFLREIGFEWVRSR'],
        ['AXE7A_XYLR2', '1', 'MFNFAPKQTTEMKKLLFTLVFVLGSMATALAENYPYRADYLWLTVPNHADWLYKTGERAKVEVSFCLYGMPQNVEVAYEIGPDMMPATSSGKVTLKNGRAVIDMGTMKKPGFLDMRLSVDGKYQHHVKVGFSPELLKPYTKNPQDFDAFWKANLDEARKTPVSVSCNKVDKYTTDAFDCYLLKIKTDRRHSIYGYLTKPKKAGKYPVVLCPPGAGIKTIKEPMRSTFYAKNGFIRLEMEIHGLNPEMTDEQFKEITTAFDYENGYLTNGLDDRDNYYMKHVYVACVRAIDYLTSLPDWDGKNVFVQGGSQGGALSLVTAGLDPRVTACVANHPALSDMAGYLDNRAGGYPHFNRLKNMFTPEKVNTMAYYDVVNFARRITCPVYITWGYNDNVCPPTTSYIVWNLITAPKESLITPINEHWTTSETNYTQMLWLKKQVK'],
        ['A0A0B8RHP0_LISMN', '2', 'MKKLLFLGDSVTDAGRDFENDRELGHGYVKIIADQLEQEDVTVINRGVSANRVADLHRRIEADAISLQPDVVTIMIGINDTWFSFSRWEDTSVTAFKEVYRVILNRIKTETNAELILMEPFVLPYPEDRKEWRGDLDPKIGAVRELAAEFGATLIPLDGLMNALAIKHGPTFLAEDGVHPTKAGHEAIASTWLEFTK']]


def ensure_esm3_weights(cache_dir=None):
    """Download the ESM3-open weights if absent, returning the snapshot path.

    Mirrors ensure_unimol_weights() in examples/encode_reaction.py, and exists for
    the same reason: the default cache lives inside the container (here
    ~/.cache/huggingface), so `docker run --rm` discards 5.4 GB and re-downloads
    it on the next run. HF_HOME is the knob, and it is set at the top of this
    module because huggingface_hub reads it at import time.

    Two differences from the UniMol case:

    * The skip-if-present logic is free. snapshot_download already re-downloads
      nothing when the local snapshot is complete, so the check below only
      decides which message to print.
    * No token is involved. EvolutionaryScale/esm3-sm-open-v1 is ungated
      (`gated: false`), so this downloads anonymously -- despite the interactive
      huggingface_hub.login() that EmbedESM3 used to call before it was removed.

    Calling this before constructing EmbedESM3 is not required, only kinder: it
    turns a 5.4 GB stall inside a constructor into a visible, resumable step.
    """
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir or os.environ['HF_HOME']) / 'hub'
    cache_dir.mkdir(parents=True, exist_ok=True)

    marker = cache_dir / f"models--{ESM3_REPO.replace('/', '--')}"
    print(f'checking for ESM3 weights:')
    if marker.exists():
        print(f'\tcached: {marker}')
    else:
        print(f'\tdownloading {ESM3_REPO} (~5.4 GB) -> {cache_dir}')

    return snapshot_download(repo_id=ESM3_REPO, cache_dir=str(cache_dir))


def main():
    df = pd.DataFrame(rows, columns=[id_col, label_col, seq_col])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Save calls to_pickle directly

    ensure_esm3_weights()
    # The step is a callable that takes a DataFrame and returns a DataFrame. 
    print("\nCalling EmbedESM3 to embed protein sequences with ESM3-open...")
    df << (EmbedESM3(id_col, seq_col) >> Save(str(OUTPUT_PKL)))

    widths = {np.asarray(v).shape for v in df['esm3_mean']}
    print(f'\tesm3_mean width and type: {widths} {np.asarray(df["esm3_mean"].iloc[0]).dtype}')
    print(f'\twrote {OUTPUT_PKL}')
    return df


if __name__ == '__main__':
    main()
