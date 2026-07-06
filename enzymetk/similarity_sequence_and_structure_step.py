from enzymetk.step import Step

import logging
import os
import pandas as pd
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import random
import string
from typing import List, Optional, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[EnzymeTK] %(message)s"))
    logger.addHandler(_handler)

class FoldSeekDatabase(str, Enum):
    # Predefined databases available via `foldseek databases` command
    # the string values must match the names used by foldseek 
    # for its built-in databases
    PDB = "PDB"
    AFDB_SWISSPROT = "Alphafold/Swiss-Prot"
    CUSTOM = "CUSTOM" # Special enum value to trigger custom DB building from reference sequences in the dataframe


class FoldSeek(Step):
    """Unified sequence and structure similarity search using FoldSeek.

    Supports two query modes:
      - **Structure mode**: provide CIF/PDB file paths via `structure_column_name`.
        Runs `foldseek easy-search` directly with the structure files.
      - **Sequence mode**: provide amino-acid sequences via `sequence_column_name`.
        Writes a FASTA, builds a query DB with ProstT5, then searches.

    The mode is determined by whether `structure_column_name` is set.

    Accepts a list of databases to search against. Expensive one-time operations
    (ProstT5 query DB creation, structure file preparation) happen once, then
    each database is searched. Results are returned as a single DataFrame with a
    ``database`` column identifying which database each hit came from.
    """

    def __init__(
        self,
        id_column_name: str,  # Column containing unique sequence/structure identifiers
        sequence_column_name: str,  # Column containing amino-acid sequences
        prostt5_weights_path: str,  # Path to ProstT5 model weights directory
        databases: List[Union[FoldSeekDatabase, str]],  # Enum values or named pre-built DB strings
        database_root_path: str,  # Root directory; sub-paths derived as root/{name}/{name}
        structure_column_name: Optional[str] = None,  # Column with CIF/PDB file paths; enables structure mode
        label_column_name: Optional[str] = None,  # Column with 'query'/'reference' labels for custom DB builds
        args: Optional[List[str]] = None,  # Extra CLI arguments passed to foldseek easy-search
        tmp_dir: Optional[str] = None,  # Custom temporary directory; defaults to system temp
        env_name: Optional[str] = None,  # Conda environment name to run foldseek commands in
    ):
        super().__init__()
        self.id_column_name = id_column_name
        self.sequence_column_name = sequence_column_name
        self.structure_column_name = structure_column_name
        self.label_column_name = label_column_name
        self.prostt5_weights_path = str(prostt5_weights_path)
        self.databases = self._classify_databases(databases)
        self.database_root_path = str(database_root_path)
        self.args = args
        self.tmp_dir = tmp_dir
        self.conda = env_name
        self.env_name = env_name

    # ------------------------------------------------------------------
    # Database classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_databases(
        databases: List[Union[FoldSeekDatabase, str]],
    ) -> List[Union[FoldSeekDatabase, str]]:
        """Classify each entry as a known FoldSeekDatabase enum or a named string.

        Enum values (including already-typed FoldSeekDatabase instances) are
        stored as-is.  Any other string is kept as a raw name and will be
        resolved as a pre-built custom database under database_root_path.
        """
        classified: List[Union[FoldSeekDatabase, str]] = []
        for db in databases:
            if isinstance(db, FoldSeekDatabase):
                classified.append(db)
                continue
            try:
                classified.append(FoldSeekDatabase(db))
            except ValueError:
                classified.append(str(db))
        return classified

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _db_prefix_for(self, db: Union[FoldSeekDatabase, str]) -> str:
        """Return the foldseek database file prefix for a given database.

        For FoldSeekDatabase enums: root/{db.name}/{db.name}
        For named strings:         root/{name}/{name}
        """
        name = db.name if isinstance(db, FoldSeekDatabase) else db
        db_dir = Path(self.database_root_path) / name
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / name)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _ensure_prostt5_weights(self):
        """Download ProstT5 weights if not already present."""
        weights_dir = Path(self.prostt5_weights_path)
        weights_dir.mkdir(parents=True, exist_ok=True)
        if not (weights_dir / "prostt5-f16.gguf").exists():
            logger.info("ProstT5 weights not found at %s — downloading...", weights_dir)
            tmp_dl = weights_dir / "tmp"
            tmp_dl.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["foldseek", "databases", "ProstT5", str(weights_dir), str(tmp_dl)],
                check=True,
            )

    def _database_exists(self, db: Union[FoldSeekDatabase, str]) -> bool:
        """Check whether foldseek database files exist for a given database."""
        db_prefix = self._db_prefix_for(db)
        return Path(f"{db_prefix}.index").exists()

    def _build_custom_database(self, reference_df: pd.DataFrame):
        """Build a custom foldseek DB from reference rows in the dataframe."""
        db_prefix = self._db_prefix_for(FoldSeekDatabase.CUSTOM)
        logger.info("Building Custom Database with %d reference sequences...", len(reference_df))
        with TemporaryDirectory() as tmp:
            fasta_path = os.path.join(tmp, "ref.fasta")
            with open(fasta_path, "w") as fasta_handle:
                for _, row in reference_df.iterrows():
                    fasta_handle.write(f">{row[self.id_column_name]}\n{row[self.sequence_column_name]}\n")
            subprocess.run(
                ["foldseek", "createdb", fasta_path, db_prefix,
                 "--prostt5-model", self.prostt5_weights_path],
                check=True,
            )
        logger.info("Custom DB created at %s", db_prefix)

    def _manage_all_databases(self, df: Optional[pd.DataFrame] = None):
        """Ensure all requested databases are available."""
        for db in self.databases:
            db_prefix = self._db_prefix_for(db)
            if self._database_exists(db):
                logger.info("Database found at %s", db_prefix)
                continue

            # Special handling for the CUSTOM enum: 
            # build from reference sequences in the dataframe if possible
            if db == FoldSeekDatabase.CUSTOM:
                if df is not None and self.label_column_name is not None:
                    reference_df = df[df[self.label_column_name] == "reference"]
                    if reference_df.empty:
                        raise ValueError(
                            f"No rows with label 'reference' found in column '{self.label_column_name}'."
                        )
                    self._build_custom_database(reference_df)
                    continue
                raise FileNotFoundError(
                    f"Custom database not found at {db_prefix}. "
                    "Provide a label_column_name with 'query'/'reference' rows, "
                    "or a valid path to an existing foldseek database."
                )
            
            # Special handling for named pre-built custom databases 
            # (plain string, not a FoldSeekDatabase enum)
            if not isinstance(db, FoldSeekDatabase):
                known_names = [e.value for e in FoldSeekDatabase]
                raise FileNotFoundError(
                    f"Database '{db}' not found at {db_prefix}. "
                    f"If this is a pre-built custom database, ensure foldseek "
                    f"DB files exist at that path (e.g. {db_prefix}.index). "
                    f"Known downloadable databases: {', '.join(known_names)}"
                )
            
            # For built-in FoldSeekDatabase enums, attempt to download if not found
            db_dir = Path(self.database_root_path)
            db_dir.mkdir(parents=True, exist_ok=True)
            tmp_dl = db_dir / "tmp"
            tmp_dl.mkdir(parents=True, exist_ok=True)

            logger.info("Downloading foldseek database '%s' to %s ...", db.value, db_prefix)
            subprocess.run(
                ["foldseek", "databases", db.value, db_prefix, str(tmp_dl)],
                check=True,
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _build_query(self, df: pd.DataFrame, tmp_dir: str) -> dict:
        """Build the query source once (expensive ProstT5 step for sequences).

        Returns a dict with 'mode' and either 'query_db' (sequence mode)
        or 'structure_files' (structure mode).
        """
        tmp_label = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        if self.structure_column_name is not None:
            structure_files = list(df[self.structure_column_name].values)
            logger.info("Structure mode — %d structure files prepared", len(structure_files))
            return {"mode": "structure", "structure_files": structure_files}

        # --- Sequence mode: build query DB with ProstT5 (expensive, done once) ---
        fasta_path = os.path.join(tmp_dir, f'{tmp_label}_seqs.fasta')
        with open(fasta_path, 'w') as fasta_handle:
            for _, row in df.iterrows():
                fasta_handle.write(f'>{row[self.id_column_name]}\n{row[self.sequence_column_name]}\n')

        logger.info("Building query database with ProstT5 from %d sequences...", len(df))
        query_db = os.path.join(tmp_dir, f'querydb_{tmp_label}')
        subcmd = [
            'foldseek', 'createdb', fasta_path, query_db,
            '--prostt5-model', self.prostt5_weights_path,
        ]
        self.run(subcmd)
        logger.info("Query database created.")
        return {"mode": "sequence", "query_db": query_db}

    def _db_label(self, db: Union[FoldSeekDatabase, str]) -> str:
        """Human-readable label for a database (used in logs and result columns)."""
        return db.value if isinstance(db, FoldSeekDatabase) else db

    def _search_single_db(self, query_source: dict, db: Union[FoldSeekDatabase, str], tmp_dir: str) -> pd.DataFrame:
        """Run foldseek easy-search against a single target database."""
        tmp_label = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        output_file = os.path.join(tmp_dir, f'{tmp_label}.txt')
        foldseek_tmp = os.path.join(tmp_dir, f'fs_tmp_{tmp_label}')
        os.makedirs(foldseek_tmp, exist_ok=True)

        db_prefix = self._db_prefix_for(db)
        db_label = self._db_label(db)
        logger.info("Searching %s against %s (%s)", query_source["mode"], db_label, db_prefix)

        if query_source["mode"] == "structure":
            cmd = ['foldseek', 'easy-search'] + query_source["structure_files"] + [
                db_prefix, output_file, foldseek_tmp
            ]
        else:
            cmd = ['foldseek', 'easy-search', query_source["query_db"],
                   db_prefix, output_file, foldseek_tmp]

        if self.args is not None:
            cmd.extend(self.args)

        self.run(cmd)

        result_df = pd.read_csv(output_file, header=None, sep='\t')
        result_df.columns = [
            'query', 'target', 'fident', 'alnlen', 'mismatch',
            'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits',
        ]
        result_df['database'] = db_label
        logger.info("Found %d hits in %s", len(result_df), db_label)
        return result_df

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = "Sequence + Structure" if self.structure_column_name else "Sequence only"
        logger.info("Starting FoldSeek pipeline in %s mode (%d input rows, %d databases)", mode, len(df), len(self.databases))
        # Ensure ProstT5 weights and all databases are available before searching
        self._ensure_prostt5_weights()

        # If a dataframe is provided, we can use it to build the custom database 
        # from reference sequences if needed. If not, we'll just check for the 
        # existence of the required databases and raise errors if they're missing.
        self._manage_all_databases(df)

        # Filter to query rows only when a label column is provided
        if self.label_column_name is not None:
            query_df = df[df[self.label_column_name] == "query"]
            logger.info("Filtered to %d query rows (from %d total)", len(query_df), len(df))
        else:
            query_df = df

        with TemporaryDirectory() as tmp_dir:
            # Use the custom temporary directory if provided
            tmp_dir = self.tmp_dir if self.tmp_dir is not None else tmp_dir
            # Build the query source (expensive ProstT5 step for sequences, done once)
            query_source = self._build_query(query_df, tmp_dir)
            results = []
            # Search each database and collect results
            for db in self.databases:
                try:
                    results.append(self._search_single_db(query_source, db, tmp_dir))
                except Exception as exc:
                    logger.error("Error searching %s: %s", self._db_label(db), exc)
                    continue
            if results:
                # Concatenate results from all databases into a single DataFrame
                return pd.concat(results, ignore_index=True)
            return pd.DataFrame()
