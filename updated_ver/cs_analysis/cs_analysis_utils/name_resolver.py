import os
from typing import Optional

import pandas as pd


class NameResolver:
    def __init__(
        self,
        drkg_nodes_path: Optional[str] = None,
        hetionet_nodes_path: Optional[str] = None,
    ):
        self.drkg = None
        self.hetn = None

        if drkg_nodes_path and os.path.exists(drkg_nodes_path):
            self.drkg = pd.read_csv(
                drkg_nodes_path,
                sep="\t",
                names=["node_num", "node_name", "node_type"],
            )

        if hetionet_nodes_path and os.path.exists(hetionet_nodes_path):
            self.hetn = pd.read_csv(hetionet_nodes_path, sep="\t")

        self._drkg_num2name = {}
        if (
            self.drkg is not None
            and "node_num" in self.drkg
            and "node_name" in self.drkg
        ):
            self._drkg_num2name = dict(self.drkg[["node_num", "node_name"]].values)

        self._hetn_id2name = {}
        if (
            self.hetn is not None
            and "id" in self.hetn.columns
            and "name" in self.hetn.columns
        ):
            self._hetn_id2name = dict(self.hetn[["id", "name"]].values)

    def drug_label(self, x: int) -> str:
        try:
            if self.drkg is not None:
                node_name = self._drkg_num2name.get(int(x), None)
                if node_name:
                    if self.hetn is not None:
                        return self._hetn_id2name.get(str(node_name), str(node_name))
                    return str(node_name)
            return str(x)
        except Exception:
            return str(x)

    def pathway_label(self, node_info) -> str:
        try:
            if isinstance(node_info, int) and self.drkg is not None:
                node_name = self._drkg_num2name.get(int(node_info), None)
            else:
                node_name = str(node_info)

            if self.hetn is not None and node_name is not None:
                return self._hetn_id2name.get(str(node_name), str(node_name))
            return str(node_name) if node_name is not None else "Unknown"
        except Exception:
            return str(node_info)
