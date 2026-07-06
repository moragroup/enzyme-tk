from enzymetk.step import Step

import logging
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
import subprocess
import random
import string
from tqdm import tqdm 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_clustering(filename, df, id_column_name, split_chain=True):
    clustering = pd.read_csv(filename, delimiter='\t', header=None)
    #rename heading as cluster reference and id
    clustering.columns = ['foldseek_representative_cluster_structure', id_column_name]
    clustering.drop_duplicates(subset=id_column_name, keep='first', inplace=True)
    if split_chain:
        # PDB-style ids look like "1abc_A"; split the chain into its own column.
        clustering['chain'] = [c.split('_')[-1] for c in clustering[id_column_name].values]
        clustering[id_column_name] = ['_'.join(c.split('_')[:-1]) for c in clustering[id_column_name].values]
    clustering.set_index(id_column_name, inplace=True)
    # Join the clustering with the df
    df = df.set_index(id_column_name)
    df = df.join(clustering, how='left')
    df.reset_index(inplace=True)
    return df

class FoldSeek(Step):
    
    def __init__(self, id_column_name: str, query_column_name: str, reference_database: str, method='search', query_type='structures', 
                 args=None, num_threads=1, tmp_dir: str = None, venv_name = 'enzymetk', env_name = None):
        super().__init__()
        self.query_column_name = query_column_name
        self.id_column_name = id_column_name
        self.reference_database = reference_database # pdb should be the default
        self.tmp_dir = tmp_dir
        self.method = method
        self.args = args
        self.num_threads = num_threads
        self.query_type = query_type
        self.conda = env_name
        self.env_name = env_name
        if self.method not in ['search', 'cluster']:
            print('Method must be in "search" or "cluster". Will likely fail... ')
        if self.query_type not in ['seqs', 'structures']:
            print('query_type must be either "seqs" or "structures" i.e. is it an amino acid sequence or a path to pdb files?')

    def __execute(self, data: list) -> np.array:
        df, tmp_dir = data
        tmp_label = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        
        if self.query_type == 'seqs':
            # Convert to a fasta
            with open(f'{tmp_dir}/{tmp_label}_seqs.fasta', 'w') as f:
                for i, row in df.iterrows():
                    f.write(f'>{row[self.id_column_name]}\n{row[self.query_column_name]}\n')
            
        # Get the PDB files from the column
        pdb_files = list(df[self.query_column_name].values)
        
        if self.method == 'search':
            cmd = ['foldseek', 'easy-search']
            if self.query_type == 'structures':
                cmd += pdb_files + [f'{self.reference_database}', f'{tmp_dir}{tmp_label}.txt', 'tmp']
            else:
                # Convert the file to a fasta and then pass that file name 
                # Make a db from the seqs
                # ToDo: make this more efficient
                subcmd = ['foldseek', 'databases', 'ProstT5', 'weights', 'tmp']
                self.run(subcmd)
                
                subcmd = ['foldseek', 'createdb', f'{tmp_dir}/{tmp_label}_seqs.fasta', f'db_{tmp_label}', '--prostt5-model', 'weights']
                self.run(subcmd)
                
                # Pass your newly created dB
                cmd += [f'db_{tmp_label}', f'{self.reference_database}', f'{tmp_dir}{tmp_label}.txt', 'tmp']

        elif self.method == 'cluster':
            cmd = ['foldseek', 'easy-cluster']
            if self.query_type == 'structures':
                cmd += pdb_files + [f'{tmp_dir}/clusterFolds', f'{tmp_dir}']
            else:
                # easy-cluster does not accept fasta input, so build a foldseek DB
                # from sequences (using ProstT5) and run the database-based `cluster`
                # workflow, then export the cluster mapping as a TSV with `createtsv`.
                subcmd = ['foldseek', 'databases', 'ProstT5', 'weights', 'tmp']
                self.run(subcmd)
                subcmd = ['foldseek', 'createdb', f'{tmp_dir}/{tmp_label}_seqs.fasta',
                          f'{tmp_dir}/db_{tmp_label}', '--prostt5-model', 'weights']
                self.run(subcmd)

                cluster_db = f'{tmp_dir}/clusterFolds'
                cmd = ['foldseek', 'cluster',
                       f'{tmp_dir}/db_{tmp_label}', cluster_db, f'{tmp_dir}']
                if self.args is not None:
                    cmd.extend(self.args)
                self.run(cmd)

                # Export the clusterDB to the TSV file the rest of the pipeline expects.
                tsv_cmd = ['foldseek', 'createtsv',
                           f'{tmp_dir}/db_{tmp_label}', f'{tmp_dir}/db_{tmp_label}',
                           cluster_db, f'{tmp_dir}/clusterFolds_cluster.tsv']
                self.run(tsv_cmd)
                # Skip the trailing self.run(cmd) below; cluster + createtsv already ran.
                cmd = None

        # add in args
        if self.args is not None and cmd is not None:
           cmd.extend(self.args)

        if cmd is not None:
            self.run(cmd)
        
        if self.method == 'search':
            df = pd.read_csv(f'{tmp_dir}{tmp_label}.txt', header=None, sep='\t')
            df.columns = ['query', 'target', 'fident', 'alnlen', 'mismatch', 
                      'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits']
        elif self.method == 'cluster':
            split_chain = self.query_type == 'structures'
            df = process_clustering(f'{tmp_dir}/clusterFolds_cluster.tsv', df,
                                    self.id_column_name, split_chain=split_chain)
            return df
        return df
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = self.tmp_dir if self.tmp_dir is not None else tmp_dir
            if self.num_threads > 1:
                output_filenames = []
                df_list = np.array_split(df, self.num_threads)
                for df_chunk in tqdm(df_list):
                    try:
                        output_filenames.append(self.__execute([df_chunk, tmp_dir]))
                    except Exception as e:
                         logger.error(f"Error in executing ESM2 model: {e}")
                         continue
                df = pd.DataFrame()
                print(output_filenames)
                for sub_df in output_filenames:
                    df = pd.concat([df, sub_df])
                return df
            
            else:
                df = self.__execute([df, tmp_dir])
                return df