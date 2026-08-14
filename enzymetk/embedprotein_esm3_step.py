# ESM 3 script
from tempfile import TemporaryDirectory
import torch
import os
import pandas as pd
from enzymetk.step import Step
import numpy as np
from tqdm import tqdm 
try:
    from esm.sdk.api import ESMProtein
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, SamplingConfig
except ImportError as e:
    # Report which module was missing. `esm` pulls in a fair few transitive
    # imports (esm/sdk/__init__ loads the Forge HTTP client before you reach
    # esm.sdk.api), so "install esm" is often the wrong advice -- esm can be
    # installed and this still fires. Without the name, the next symptom is a
    # NameError on ESM3 much further down, which points nowhere useful.
    print(f"EmbedESM3: import failed ({e}). Needs the esm3 package and its "
          f"dependencies. Install with: pip install esm.")

# CUDA setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152


def resolve_device(device=None):
    """Pick the torch device to run ESM3 on.

    Explicit argument wins, then the ESM3_DEVICE env var, then whatever is
    actually available. Previously this module hard-coded cuda, which made the
    step unconstructable on any CPU-only host (containers, laptops, CI).
    """
    name = device or os.environ.get('ESM3_DEVICE')
    if not name:
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


class EmbedESM3(Step):

    def __init__(self, id_col: str, seq_col: str, extraction_method='mean', num_threads=1,
                 tmp_dir: str = None, env_name: str = 'enzymetk', save_tensors=False,
                 device: str = None): # type: ignore
        # No login() here: EvolutionaryScale/esm3-sm-open-v1 is ungated, so the
        # weights download anonymously. A bare login() prompts interactively and
        # hangs anywhere without a TTY. huggingface_hub still picks up HF_TOKEN
        # from the environment on its own for anyone pointing at a private mirror.
        self.device = resolve_device(device)
        # from_pretrained takes the device and casts to bfloat16 only off-CPU;
        # a trailing .to("cuda") would override that and raise without a GPU.
        self.client = ESM3.from_pretrained("esm3-open", device=self.device)
        self.seq_col = seq_col
        self.id_col = id_col
        self.num_threads = num_threads or 1
        self.extraction_method = extraction_method
        self.tmp_dir = tmp_dir
        self.env_name = env_name
        self.save_tensors = save_tensors

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> pd.DataFrame: 
        client = self.client
        means = []
        for id, seq in tqdm(df[[self.id_col, self.seq_col]].values):
            protein = ESMProtein(
                sequence=(
                    seq
                )
            )
            protein_tensor = client.encode(protein)
            output = client.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            if self.save_tensors:
                torch.save(output.per_residue_embedding, os.path.join(tmp_dir, f'{id}.pt'))
            # .float() before numpy: off-CPU the model runs in bfloat16, which
            # numpy cannot convert. On CPU it is already float32 and this is a no-op.
            means.append(np.array(output.per_residue_embedding.mean(dim=0).float().cpu()))
        df['esm3_mean']  = means
        return df
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.tmp_dir is None:
            with TemporaryDirectory() as tmp_dir:
                if self.num_threads > 1:
                    dfs = []
                    df_list = np.array_split(df, self.num_threads)
                    for df_chunk in tqdm(df_list):
                        dfs.append(self.__execute(df_chunk, tmp_dir))
                    df = pd.DataFrame()
                    for tmp_df in tqdm(dfs):
                        df = pd.concat([df, tmp_df])
                    return df
                else:
                    df = self.__execute(df, tmp_dir)
                    return df
        else:
            df = self.__execute(df, self.tmp_dir)
            return df
