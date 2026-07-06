import pandas as pd
from tempfile import TemporaryDirectory
import logging
import numpy as np
from enzymetk.step import Step
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DRFP(Step):
    
    def __init__(self, smiles_col: str, num_threads = 1,
                 env_name = 'enzymetk', venv_name = None, tmp_dir = None):
        super().__init__()
        self.smiles_col = smiles_col
        self.num_threads = num_threads
        self.conda = env_name
        self.env_name = env_name
        self.tmp_dir = tmp_dir
        self.venv = venv_name if venv_name else f'{env_name}/bin/python'
        
    def install(self, env_args=None):
        self.u.err_p("To install, see the wiki! Or go to conda_envs/drfp.sh and run the commands there.")

    def __execute(self, df: pd.DataFrame) -> pd.DataFrame:
        from drfp import DrfpEncoder
        smiles_list = list(df[self.smiles_col].values)
        encodings = []
        count_failed = 0    
        # encode them like this so that if it failes on one smiles string, it doesn't fail on the whole batch
        for smiles in tqdm(smiles_list, desc="Encoding SMILES"):
            try:
                encodings.append(DrfpEncoder.encode(smiles))
            except Exception as e:
                count_failed += 1
                encodings.append(None)
        df['drfp_fps']  = encodings
        logger.info(f"Failed to encode {count_failed} SMILES strings. Successfully encoded {len(smiles_list) - count_failed} SMILES strings.")
        print([f"Failed to encode {count_failed} SMILES strings. Successfully encoded {len(smiles_list) - count_failed} SMILES strings."])
        return df
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.tmp_dir:
            with TemporaryDirectory() as tmp_dir:
                self.tmp_dir = tmp_dir
        if self.num_threads > 1:
            data = []
            df_list = np.array_split(df, self.num_threads)
            for df_chunk in df_list:
                data.append(df_chunk)
            pool = ThreadPool(self.num_threads)
            output_filenames = pool.map(self.__execute, data)
            df = pd.DataFrame()
            for tmp_df in output_filenames:
                df = pd.concat([df, tmp_df])
            return df
        else:
            return self.__execute(df)
                
                