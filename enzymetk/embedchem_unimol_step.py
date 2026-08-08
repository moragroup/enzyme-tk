import os
import pandas as pd
from tempfile import TemporaryDirectory
import logging
import numpy as np
from enzymetk.step import Step

from multiprocessing.dummy import Pool as ThreadPool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class UniMol(Step):
    
    def __init__(self, smiles_col: str, unimol_model = 'unimolv2',
                 unimol_size = '164m', num_threads = 1,
                 env_name = 'enzymetk', venv_name = None,
                 weights_dir = None):
        # weights_dir: directory holding an already-downloaded UniMol checkpoint, laid out
        # as unimol_tools expects (unimolv2/164m -> modelzoo/164M/checkpoint.pt).  Left None,
        # unimol_tools resolves it from $UNIMOL_WEIGHT_DIR as before.  Added last in the
        # signature so existing positional calls keep working.
        super().__init__()
        self.smiles_col = smiles_col
        self.num_threads = num_threads
        self.conda = env_name
        self.env_name = env_name
        self.venv = venv_name if venv_name else f'{env_name}/bin/python'
        self.unimol_model = unimol_model
        self.unimol_size = unimol_size
        self.weights_dir = weights_dir
        
 
    def install(self, env_args=None):
        # e.g. env args could by python=='3.1.1.
        self.install_venv(env_args)
        # Now the specific
        try:
            cmd = [f'{self.env_name}/bin/pip', 'install', 'unimol_tools']
            self.run(cmd)
        except Exception as e:
            cmd = [f'{self.env_name}/bin/pip3', 'install', 'unimol_tools']
            self.run(cmd)
        self.run(cmd)
        # Now set the venv to be the location:
        self.venv = f'{self.env_name}/bin/python'

    def __execute(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_list = list(df[self.smiles_col].values)
        reprs = []
        for smile in smiles_list:
            try:
                unimol_repr = self.clf.get_repr([smile], return_atomic_reprs=True)
                reprs.append(unimol_repr['cls_repr'])
            except Exception as e:
                logger.warning(f"Error embedding smile {smile}: {e}")
                reprs.append(None)
        df['unimol_repr']  = reprs
        return df
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from unimol_tools import UniMolRepr
            # MODEL_CONFIG_V2 is only available in unimol_tools >= 0.1.3, 
            # so this import will fail if the installed version is older.
            from unimol_tools.config import MODEL_CONFIG_V2
        except ImportError as e:
            raise ImportError(
                "UniMolRepr requires unimol-tools. "
                "Install after initializing class with install()"
            ) from e
        model_name = self.unimol_model or 'unimolv2' # avaliable: unimolv1, unimolv2
        model_size = self.unimol_size or '164m' # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
        # Naming the checkpoint outright stops unimol_tools falling back to get_weight_dir(),
        # which defaults to its own site-packages folder and downloads ~660 MB into it -- once
        # per fresh container.
        pretrained_model_path = None
        if self.weights_dir and model_name != 'unimolv2':
            # Refuse rather than ignore: v1 also needs a dictionary file resolved the same
            # way, so honouring half of weights_dir would download the rest behind the
            # caller's back -- the exact thing they passed it to prevent.
            raise ValueError(
                f'weights_dir is only supported for unimolv2, not {model_name!r}. '
                'Omit weights_dir to let unimol_tools resolve the checkpoint and '
                'dictionary from $UNIMOL_WEIGHT_DIR.'
            )
        if self.weights_dir:
            # verify the model size is known, 
            # and build the path to the checkpoint unimol_tools expects
            relative_path = MODEL_CONFIG_V2['weight'].get(model_size)
            if relative_path is None:
                raise ValueError(
                    f'Unknown unimol_size {model_size!r}; expected one of '
                    f'{sorted(MODEL_CONFIG_V2["weight"])}'
                )
            pretrained_model_path = os.path.join(self.weights_dir, relative_path)
            # Naming the path also opts out of the existence check unimol_tools runs for
            # itself -- and of the download that would heal a miss.  So check here, where
            # the expected layout can still be named; torch.load would only report the file.
            if not os.path.isfile(pretrained_model_path):
                raise FileNotFoundError(
                    f'UniMol {model_size} checkpoint not found at {pretrained_model_path}. '
                    f'weights_dir must contain {relative_path!r}; omit weights_dir to let '
                    'unimol_tools download it instead.'
                )
        # single smiles unimol representation
        clf = UniMolRepr(data_type='molecule',
                        remove_hs=False,
                        model_name= model_name,
                        model_size= model_size,
                        pretrained_model_path=pretrained_model_path,
                        )
        self.clf = clf
        with TemporaryDirectory() as tmp_dir:
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
                