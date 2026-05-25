"""
Agent tool wrappers for embedding steps: ESM, ESM3, ChemBERT, RxnFP, SelFormer, UniMol.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from enzymetk.agent.registry import ToolRegistry
from enzymetk.agent.tools.base import EnzymeTool
from enzymetk.agent.schemas import ToolResult


@ToolRegistry.register
class EmbedProteinESMTool(EnzymeTool):
    """Generate protein embeddings using ESM-2."""

    name = "embed_protein_esm"
    description = (
        "Generate protein sequence embeddings using the ESM-2 language model. "
        "Supports mean pooling or active-site-specific extraction. Useful for "
        "sequence similarity, clustering, or as features for downstream ML models."
    )
    category = "embedding"
    required_env = "enzymetk"

    def run(
        self,
        sequences: List[Dict[str, str]],
        id_column: str = "Entry",
        sequence_column: str = "Sequence",
        model: str = "esm2_t36_3B_UR50D",
        extraction_method: str = "mean",
        active_site_column: Optional[str] = None,
        active_sites: Optional[List[str]] = None,
        num_threads: int = 1,
        tmp_dir: Optional[str] = None,
        rep_num: int = -1,
    ) -> ToolResult:
        from enzymetk.embedprotein_esm_step import EmbedESM

        df = self.sequences_to_df(sequences, id_column, sequence_column)
        if active_sites is not None and active_site_column:
            df[active_site_column] = active_sites

        step = EmbedESM(
            id_col=id_column,
            seq_col=sequence_column,
            model=model,
            extraction_method=extraction_method,
            active_site_col=active_site_column,
            num_threads=num_threads,
            tmp_dir=tmp_dir,
            rep_num=rep_num,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"ESM-2 ({model}) embeddings generated for {len(sequences)} sequences "
            f"using {extraction_method} extraction",
        )


@ToolRegistry.register
class EmbedProteinESM3Tool(EnzymeTool):
    """Generate protein embeddings using ESM3."""

    name = "embed_protein_esm3"
    description = (
        "Generate protein sequence embeddings using the ESM3 language model on GPU. "
        "Produces mean per-residue representations."
    )
    category = "embedding"
    required_env = "enzymetk"

    def run(
        self,
        sequences: List[Dict[str, str]],
        id_column: str = "Entry",
        sequence_column: str = "Sequence",
        extraction_method: str = "mean",
        num_threads: int = 1,
        tmp_dir: Optional[str] = None,
        save_tensors: bool = False,
    ) -> ToolResult:
        from enzymetk.embedprotein_esm3_step import EmbedESM3

        df = self.sequences_to_df(sequences, id_column, sequence_column)
        step = EmbedESM3(
            id_col=id_column,
            seq_col=sequence_column,
            extraction_method=extraction_method,
            num_threads=num_threads,
            tmp_dir=tmp_dir,
            save_tensors=save_tensors,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"ESM3 embeddings generated for {len(sequences)} sequences",
        )


@ToolRegistry.register
class EmbedChemBERTTool(EnzymeTool):
    """Generate molecular embeddings from SMILES using ChemBERTa."""

    name = "embed_molecule_chemberta"
    description = (
        "Generate molecular embeddings from SMILES strings using the ChemBERTa "
        "(PubChem10M) transformer model. Useful for molecular similarity and "
        "property prediction."
    )
    category = "embedding"
    required_env = "enzymetk"

    def run(
        self,
        smiles: List[str],
        ids: Optional[List[str]] = None,
        id_column: str = "Entry",
        smiles_column: str = "SMILES",
        num_threads: int = 1,
    ) -> ToolResult:
        from enzymetk.embedchem_chemberta_step import ChemBERT

        df = self.molecules_to_df(smiles, ids, id_column, smiles_column)
        step = ChemBERT(
            id_col=id_column,
            value_col=smiles_column,
            num_threads=num_threads,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"ChemBERTa embeddings generated for {len(smiles)} molecules",
        )


@ToolRegistry.register
class EmbedReactionDRFPTool(EnzymeTool):
    """Generate reaction embeddings using Differential Reaction Fingerprints (DRFP)."""

    name = "embed_reaction_drfp"
    description = (
        "Generate differential reaction fingerprints (DRFP) from reaction SMILES. "
        "DRFP encodes the change in molecular substructures between reactants and "
        "products, making it useful for reaction similarity search, classification, "
        "and yield prediction."
    )
    category = "embedding"
    required_env = "drfp"

    def run(
        self,
        smiles: List[str],
        ids: Optional[List[str]] = None,
        id_column: str = "Entry",
        smiles_column: str = "SMILES",
        n_folded_length: int = 2048,
        radius: int = 3,
        rings: bool = True,
        min_radius: int = 0,
        num_threads: int = 1,
        tmp_dir: Optional[str] = None,
    ) -> ToolResult:
        from enzymetk.embedchem_drfp_step import DRFP

        df = self.molecules_to_df(smiles, ids, id_column, smiles_column)
        step = DRFP(
            smiles_col=smiles_column,
            num_threads=num_threads,
            n_folded_length=n_folded_length,
            radius=radius,
            rings=rings,
            min_radius=min_radius,
            tmp_dir=tmp_dir,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"DRFP embeddings generated for {len(smiles)} reactions "
            f"(length={n_folded_length}, radius={radius})",
        )


@ToolRegistry.register
class EmbedRxnFPTool(EnzymeTool):
    """Generate reaction fingerprint embeddings using RxnFP."""

    name = "embed_reaction_rxnfp"
    description = (
        "Generate reaction fingerprint embeddings from reaction SMILES using the "
        "RxnFP model. Useful for reaction similarity search and classification."
    )
    category = "embedding"
    required_env = "rxnfp"

    def run(
        self,
        smiles: List[str],
        ids: Optional[List[str]] = None,
        id_column: str = "Entry",
        smiles_column: str = "SMILES",
        num_threads: int = 1,
        tmp_dir: Optional[str] = None,
    ) -> ToolResult:
        from enzymetk.embedchem_rxnfp_step import RxnFP

        df = self.molecules_to_df(smiles, ids, id_column, smiles_column)
        step = RxnFP(
            smiles_col=smiles_column,
            num_threads=num_threads,
            tmp_dir=tmp_dir,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"RxnFP embeddings generated for {len(smiles)} reactions",
        )


@ToolRegistry.register
class EmbedSelFormerTool(EnzymeTool):
    """Generate molecular embeddings using SELFormer."""

    name = "embed_molecule_selformer"
    description = (
        "Generate molecular embeddings from SMILES using the SELFormer model. "
        "Requires SELFormer installation directory and model weights file."
    )
    category = "embedding"
    required_env = "enzymetk"

    def run(
        self,
        smiles: List[str],
        selformer_dir: str,
        model_file: str,
        ids: Optional[List[str]] = None,
        id_column: str = "Entry",
        smiles_column: str = "SMILES",
    ) -> ToolResult:
        from enzymetk.embedchem_selformer_step import SelFormer

        df = self.molecules_to_df(smiles, ids, id_column, smiles_column)
        step = SelFormer(
            value_col=smiles_column,
            id_col=id_column,
            selformer_dir=selformer_dir,
            model_file=model_file,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"SELFormer embeddings generated for {len(smiles)} molecules",
        )


@ToolRegistry.register
class EmbedUniMolTool(EnzymeTool):
    """Generate molecular embeddings using UniMol."""

    name = "embed_molecule_unimol"
    description = (
        "Generate molecular embeddings from SMILES using UniMol (v1/v2) "
        "universal molecular representation learning."
    )
    category = "embedding"
    required_env = "enzymetk"

    def run(
        self,
        smiles: List[str],
        ids: Optional[List[str]] = None,
        id_column: str = "Entry",
        smiles_column: str = "SMILES",
        unimol_model: str = "unimolv2",
        unimol_size: str = "164m",
        num_threads: int = 1,
    ) -> ToolResult:
        from enzymetk.embedchem_unimol_step import UniMol

        df = self.molecules_to_df(smiles, ids, id_column, smiles_column)
        step = UniMol(
            smiles_col=smiles_column,
            unimol_model=unimol_model,
            unimol_size=unimol_size,
            num_threads=num_threads,
        )
        result_df = step.execute(df)
        return self.make_result(
            result_df,
            self.name,
            f"UniMol ({unimol_model}) embeddings generated for {len(smiles)} molecules",
        )
