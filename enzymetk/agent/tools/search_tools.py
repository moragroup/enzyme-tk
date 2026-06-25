"""
Agent tool wrappers for search/similarity steps:
BLAST, MMseqs, FoldSeek, StructureFoldSeek, ReactionDist, SubstrateDist.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import pandas as pd

from enzymetk.agent.registry import ToolRegistry
from enzymetk.agent.tools.base import EnzymeTool
from enzymetk.agent.schemas import ToolResult


@ToolRegistry.register
class SearchBlastTool(EnzymeTool):
    """Search sequences using Diamond BLAST."""

    name = "search_blast"
    description = (
        "Search protein sequences against a reference database or set of reference "
        "sequences using Diamond BLAST. Returns hits with sequence identity, "
        "e-values, and bitscores. Use for homology search."
    )
    category = "search"
    required_env = "enzymetk"

    def run(
        self,
        sequences: List[Dict[str, str]],
        labels: Optional[List[str]] = None,
        database: Optional[str] = None,
        id_column: str = "Entry",
        sequence_column: str = "Sequence",
        label_column: str = "label",
        mode: str = "blastp",
        args: Optional[List[str]] = None,
        tmp_dir: Optional[str] = None,
        num_threads: int = 1,
    ) -> ToolResult:
        from enzymetk.sequence_search_blast import BLAST

        if labels is not None:
            df = self.labelled_sequences_to_df(
                sequences, labels, id_column, label_column, sequence_column
            )
            lbl_col = label_column
        else:
            df = self.sequences_to_df(sequences, id_column, sequence_column)
            lbl_col = None

        step = BLAST(
            id_col=id_column,
            sequence_col=sequence_column,
            label_col=lbl_col,
            database=database,
            mode=mode,
            args=args,
            tmp_dir=tmp_dir,
            num_threads=num_threads,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"BLAST found {len(result_df)} hits for {len(sequences)} query sequences",
        )


@ToolRegistry.register
class SearchMMseqsTool(EnzymeTool):
    """Search or cluster sequences using MMseqs2."""

    name = "search_mmseqs"
    description = (
        "Search protein sequences against a database or cluster them using MMseqs2. "
        "Supports 'search' (all-vs-all or vs reference DB) and 'cluster' modes. "
        "Very fast for large-scale sequence comparison."
    )
    category = "search"
    required_env = "enzymetk"

    def run(
        self,
        sequences: List[Dict[str, str]],
        method: str = "search",
        reference_database: Optional[str] = None,
        id_column: str = "Entry",
        sequence_column: str = "Sequence",
        tmp_dir: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> ToolResult:
        from enzymetk.similarity_mmseqs_step import MMseqs

        df = self.sequences_to_df(sequences, id_column, sequence_column)
        step = MMseqs(
            id_column_name=id_column,
            seq_column_name=sequence_column,
            method=method,
            reference_database=reference_database,
            tmp_dir=tmp_dir,
            args=args,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"MMseqs2 {method} completed for {len(sequences)} sequences "
            f"({len(result_df)} results)",
        )


@ToolRegistry.register
class SearchFoldSeekTool(EnzymeTool):
    """Search or cluster by structural similarity using FoldSeek."""

    name = "search_foldseek"
    description = (
        "Search or cluster protein structures/sequences by structural similarity "
        "using FoldSeek. Supports 'search' and 'cluster' modes against a reference "
        "structure database."
    )
    category = "search"
    required_env = "enzymetk"

    def run(
        self,
        query_paths: List[str],
        ids: List[str],
        reference_database: str,
        method: str = "search",
        query_type: str = "structures",
        id_column: str = "Entry",
        query_column: str = "query",
        tmp_dir: Optional[str] = None,
        args: Optional[List[str]] = None,
        num_threads: int = 1,
    ) -> ToolResult:
        from enzymetk.similarity_foldseek_step import FoldSeek

        df = pd.DataFrame({id_column: ids, query_column: query_paths})
        step = FoldSeek(
            id_column_name=id_column,
            query_column_name=query_column,
            reference_database=reference_database,
            method=method,
            query_type=query_type,
            args=args,
            num_threads=num_threads,
            tmp_dir=tmp_dir,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"FoldSeek {method} completed for {len(ids)} queries",
        )


@ToolRegistry.register
class StructureSearchFoldSeekTool(EnzymeTool):
    """Search a structure database with PDB query files using FoldSeek."""

    name = "structure_search_foldseek"
    description = (
        "Search a reference structure database with PDB query files using FoldSeek "
        "easy-search. Returns structural similarity hits."
    )
    category = "search"
    required_env = "enzymetk"

    def run(
        self,
        query_paths: List[str],
        reference_database_directory: str,
        query_column: str = "pdbs",
        tmp_dir: Optional[str] = None,
        args: Optional[List[str]] = None,
        num_threads: int = 1,
    ) -> ToolResult:
        from enzymetk.structure_search_foldseek import FoldSeek as StructFoldSeek

        df = pd.DataFrame({query_column: query_paths})
        step = StructFoldSeek(
            query_column_name=query_column,
            reference_database_directory=reference_database_directory,
            args=args,
            num_threads=num_threads,
            tmp_dir=tmp_dir,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"Structure search completed for {len(query_paths)} queries",
        )


@ToolRegistry.register
class SequenceStructureSearchFoldSeekTool(EnzymeTool):
    """Unified sequence + structure similarity search using FoldSeek with ProstT5."""

    name = "search_foldseek_seq_struct"
    description = (
        "Search protein sequences or structures against a reference database "
        "(PDB, AlphaFold/Swiss-Prot, or custom) using FoldSeek. Sequences are "
        "converted to structural tokens via ProstT5; structures are searched directly. "
        "Returns hits with fractional identity, e-values, and bitscores."
    )
    category = "search"
    required_env = "enzymetk"

    def run(
        self,
        sequences: List[Dict[str, str]],
        prostt5_weights_path: str,
        database_root_path: str,
        databases: List[str] = None,
        structure_paths: Optional[List[str]] = None,
        id_column: str = "Entry",
        sequence_column: str = "Sequence",
        structure_column: str = "structure",
        args: Optional[List[str]] = None,
        num_threads: int = 1,
        tmp_dir: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> ToolResult:
        from enzymetk.similarity_sequence_and_structure_step import (
            FoldSeek as SeqStructFoldSeek,
            FoldSeekDatabase,
        )

        if databases is None:
            databases = ["PDB"]

        df = self.sequences_to_df(sequences, id_column, sequence_column)

        struct_col = None
        if structure_paths is not None:
            df[structure_column] = structure_paths
            struct_col = structure_column

        step = SeqStructFoldSeek(
            id_column_name=id_column,
            sequence_column_name=sequence_column,
            prostt5_weights_path=prostt5_weights_path,
            databases=databases,
            database_root_path=database_root_path,
            structure_column_name=struct_col,
            args=args,
            num_threads=num_threads,
            tmp_dir=tmp_dir,
            env_name=env_name,
        )
        result_df = step.execute(df)
        mode = "structure" if struct_col else "sequence"
        return self.make_result(
            result_df,
            self.name,
            f"FoldSeek {mode} search completed: {len(result_df)} hits for {len(sequences)} queries across {len(databases)} databases",
        )


@ToolRegistry.register
class ReactionSimilarityTool(EnzymeTool):
    """Compute reaction SMARTS fingerprint similarity."""

    name = "similarity_reaction"
    description = (
        "Compute Tanimoto/Russel/Cosine similarity between reaction SMARTS "
        "fingerprints using RDKit. Compares reactions in the DataFrame against "
        "a query reaction SMILES."
    )
    category = "search"
    required_env = "enzymetk"

    def run(
        self,
        smiles: List[str],
        query_smiles: str,
        ids: Optional[List[str]] = None,
        id_column: str = "Entry",
        smiles_column: str = "SMILES",
        num_threads: int = 1,
    ) -> ToolResult:
        from enzymetk.similarity_reaction_step import ReactionDist

        df = self.molecules_to_df(smiles, ids, id_column, smiles_column)
        step = ReactionDist(
            id_column_name=id_column,
            smiles_column_name=smiles_column,
            smiles_string=query_smiles,
            num_threads=num_threads,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"Reaction similarity computed for {len(smiles)} reactions",
        )


@ToolRegistry.register
class SubstrateSimilarityTool(EnzymeTool):
    """Compute substrate Morgan fingerprint similarity."""

    name = "similarity_substrate"
    description = (
        "Compute Tanimoto/Russel/Cosine similarity between substrate Morgan "
        "fingerprints using RDKit. Compares substrates in the DataFrame against "
        "a query substrate SMILES."
    )
    category = "search"
    required_env = "enzymetk"

    def run(
        self,
        smiles: List[str],
        query_smiles: str,
        ids: Optional[List[str]] = None,
        id_column: str = "Entry",
        smiles_column: str = "SMILES",
        num_threads: int = 1,
    ) -> ToolResult:
        from enzymetk.similarity_substrate_step import SubstrateDist

        df = self.molecules_to_df(smiles, ids, id_column, smiles_column)
        step = SubstrateDist(
            id_column_name=id_column,
            smiles_column_name=smiles_column,
            smiles_string=query_smiles,
            num_threads=num_threads,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"Substrate similarity computed for {len(smiles)} molecules",
        )
