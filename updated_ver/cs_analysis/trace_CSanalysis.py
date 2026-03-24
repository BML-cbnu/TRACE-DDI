"""
trace_CSanalysis.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")


DDI_PATH_DEFAULT = "./ddi/data/tsv/ddi_pairs.tsv"
SUBG_BASE_DEFAULT = "./ddi/data/subgraphs/preprocessed_kg_subgraphs"
SAVE_ROOT_DEFAULT = "./ddi/data/results"
DRKG_NODES_DEFAULT = "./ddi/data/tsv/drkg_nodes.tsv"
HETIONET_NODES_DEFAULT = "./ddi/data/tsv/hetionet_filtered_nodes.tsv"


@dataclass(frozen=True)
class PipelineArgs:
    ddi: str
    subg_base: str
    save_root: str
    drkg_nodes: str
    hetionet_nodes: str
    save_enhanced_graphs: bool
    enhanced_max_labels: int


def main(args: PipelineArgs) -> int:
    from cs_analysis_utils.pipeline_runner import run_pipeline

    return run_pipeline(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BML-DDI pipeline (modularized)")
    parser.add_argument("--ddi", default=DDI_PATH_DEFAULT, help="Path to DDI TSV")
    parser.add_argument(
        "--subg_base", default=SUBG_BASE_DEFAULT, help="Base directory of subgraphs"
    )
    parser.add_argument(
        "--save_root",
        default=SAVE_ROOT_DEFAULT,
        help="Root directory to save all results",
    )
    parser.add_argument(
        "--drkg_nodes", default=DRKG_NODES_DEFAULT, help="DRKG nodes.tsv"
    )
    parser.add_argument(
        "--hetionet_nodes", default=HETIONET_NODES_DEFAULT, help="Hetionet nodes.tsv"
    )
    parser.add_argument(
        "--save_enhanced_graphs",
        action="store_true",
        help="Save enhanced graph visualizations in addition to default outputs",
    )
    parser.add_argument(
        "--enhanced_max_labels",
        type=int,
        default=25,
        help="Maximum number of labels in enhanced graph visualization",
    )
    ns = parser.parse_args()

    args = PipelineArgs(
        ddi=str(ns.ddi),
        subg_base=str(ns.subg_base),
        save_root=str(ns.save_root),
        drkg_nodes=str(ns.drkg_nodes),
        hetionet_nodes=str(ns.hetionet_nodes),
        save_enhanced_graphs=bool(ns.save_enhanced_graphs),
        enhanced_max_labels=int(ns.enhanced_max_labels),
    )
    sys.exit(main(args))
