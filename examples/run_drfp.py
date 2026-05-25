from enzymetk.embedchem_drfp_step import DRFP
from enzymetk.save_step import Save
import pandas as pd

output_dir = 'tmp/'
num_threads = 1
id_col = 'Entry'
seq_col = 'Sequence'
substrate_col = 'Substrate'
rows = [['P0DP23', 'MALWMRLLPLLALLALWGPDPAAAMALWMRLLPLLALLALWGPDPAAAMALWMRLLPLLALLALWGPDPAAA', 'CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC>>O=C(O)C1=CC=CC=C1C(O)=O'], 
        ['P0DP24', 'MALWMRLLPLLALLALWGPDPAAAMALWMRLLPLLALLALWGPDPAAAMALWMRLLPLLALLALWGPDPAAA', 'O=C(OC(C)C)NC1=CC=CC(Cl)=C1>>O=C(O)C1=CC=CC=C1C(O)=O']]
df = pd.DataFrame(rows, columns=[id_col, seq_col, substrate_col])
df = df << (DRFP(substrate_col) >> Save(f'{output_dir}drfp.pkl'))
print(df)
