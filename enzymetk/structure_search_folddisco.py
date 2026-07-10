from enzymetk.step import Step
import logging
import os
import pandas as pd
from io import StringIO
from tempfile import TemporaryDirectory
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Output column names for the two folddisco display granularities. Assigned after
# reading folddisco's tab-separated stdout (which carries no header unless --header).
PER_MATCH_COLS = ['tid', 'node_count', 'idf', 'rmsd', 'matching_residues', 'query_residues']
PER_STRUCTURE_COLS = ['tid', 'idf', 'total_match_count', 'node_count', 'edge_count',
                      'max_node_cov', 'min_rmsd', 'nres', 'plddt', 'matching_residues',
                      'db_key', 'query_residues']


class Folddisco(Step):
    """Search discontinuous structural motifs against a prebuilt folddisco index.

    Motif-search sibling of FoldSeek: for each query row it runs `folddisco query`
    against a prebuilt geometric-hash index. Build an index once with build_index().
    """

    def __init__(self, id_col: str, query_column_name: str, index_path: str,
                 motif_col: str = None, per_structure: bool = False, header: bool = False,
                 args=None, num_threads: int = 1, tmp_dir: str = None, env_name: str = None):
        # id_col            : identifier per query row (added to output)
        # query_column_name : column holding path to query PDB/CIF file
        # index_path        : prebuilt folddisco index (reference DB) prefix
        # motif_col         : column with motif-residue string; None => whole-structure search
        # per_structure     : True => `--per-structure` output granularity
        # header            : True => folddisco emits a header row (--header)
        super().__init__()
        self.id_col = id_col
        self.query_column_name = query_column_name
        self.index_path = index_path
        self.motif_col = motif_col
        self.per_structure = per_structure
        self.header = header
        self.args = args
        self.num_threads = num_threads
        self.tmp_dir = tmp_dir
        self.venv = None
        self.conda = env_name  # None => run folddisco from PATH

    def build_index(self, structure_dir: str, index_path: str, args=None):
        """One-off DB build (not called by execute()). Mirrors FoldSeek.make_database()."""
        # folddisco writes the index prefix into an existing directory; it won't
        # create the parent itself, so ensure it exists first.
        parent = os.path.dirname(index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        cmd = ['folddisco', 'index', '-p', structure_dir, '-i', index_path,
               '-t', str(self.num_threads)]
        if args:
            cmd += args
        self.run(cmd)
        return index_path

    def __columns(self):
        return PER_STRUCTURE_COLS if self.per_structure else PER_MATCH_COLS

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = self.tmp_dir if self.tmp_dir is not None else tmp_dir
            columns = self.__columns()
            results = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                cmd = ['folddisco', 'query', '-i', self.index_path,
                       '-p', row[self.query_column_name]]
                motif = row[self.motif_col] if self.motif_col is not None else None
                if motif is not None and not pd.isna(motif) and str(motif).strip():
                    cmd += ['-q', str(motif)]
                if self.per_structure:
                    cmd += ['--per-structure']
                if self.header:
                    cmd += ['--header']
                cmd += ['-t', str(self.num_threads)]
                if self.args is not None:
                    cmd.extend(self.args)

                result = self.run(cmd)
                stdout = result.stdout
                if stdout is None or not stdout.strip():
                    # No hits for this query -> empty frame with the expected columns.
                    hits = pd.DataFrame(columns=columns)
                else:
                    hits = pd.read_csv(StringIO(stdout), sep='\t',
                                       header=0 if self.header else None)
                    hits.columns = columns
                hits[self.id_col] = row[self.id_col]
                results.append(hits)

            if not results:
                out = pd.DataFrame(columns=columns + [self.id_col])
            else:
                out = pd.concat(results, ignore_index=True)
            return out
