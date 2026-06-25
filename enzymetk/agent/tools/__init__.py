"""
Enzyme tool wrappers that bridge the existing Step classes to the agent framework.

Importing this package registers all tools with the ToolRegistry.
"""

from enzymetk.agent.tools.annotation_tools import (
    AnnotateECCleanTool,
    AnnotateECCreepTool,
    AnnotateECProteInferTool,
)
from enzymetk.agent.tools.docking_tools import (
    DockBoltzTool,
    DockChaiTool,
    DockVinaTool,
)
from enzymetk.agent.tools.embedding_tools import (
    EmbedProteinESMTool,
    EmbedProteinESM3Tool,
    EmbedChemBERTTool,
    EmbedRxnFPTool,
    EmbedSelFormerTool,
    EmbedUniMolTool,
)
from enzymetk.agent.tools.search_tools import (
    SearchBlastTool,
    SearchMMseqsTool,
    SearchFoldSeekTool,
    StructureSearchFoldSeekTool,
    SequenceStructureSearchFoldSeekTool,
    ReactionSimilarityTool,
    SubstrateSimilarityTool,
)
from enzymetk.agent.tools.alignment_tools import (
    ClustalOmegaTool,
    FastTreeTool,
)
from enzymetk.agent.tools.design_tools import (
    LigandMPNNTool,
)
from enzymetk.agent.tools.prediction_tools import (
    ActiveSitePredTool,
)
from enzymetk.agent.tools.metagenomics_tools import (
    PoreChopTool,
    ProkkaTool,
)

__all__ = [
    # Annotation
    "AnnotateECCleanTool", "AnnotateECCreepTool", "AnnotateECProteInferTool",
    # Docking
    "DockBoltzTool", "DockChaiTool", "DockVinaTool",
    # Embedding
    "EmbedProteinESMTool", "EmbedProteinESM3Tool",
    "EmbedChemBERTTool", "EmbedRxnFPTool", "EmbedSelFormerTool", "EmbedUniMolTool",
    # Search
    "SearchBlastTool", "SearchMMseqsTool", "SearchFoldSeekTool",
    "StructureSearchFoldSeekTool", "SequenceStructureSearchFoldSeekTool",
    "ReactionSimilarityTool", "SubstrateSimilarityTool",
    # Alignment
    "ClustalOmegaTool", "FastTreeTool",
    # Design
    "LigandMPNNTool",
    # Prediction
    "ActiveSitePredTool",
    # Metagenomics
    "PoreChopTool", "ProkkaTool",
]
