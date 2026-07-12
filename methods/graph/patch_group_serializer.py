from __future__ import annotations

import json
from pathlib import Path

from utils.logger import get_logger


class PatchGroupSerializer:
    """
    Serializes PatchGroup objects to disk.

    The serializer stores only graph connectivity information.
    Patch images are never duplicated. During training, patch
    images are loaded from disk using the stored node ids.

    Output
    ------
    groups.json
    """

    def __init__(
        self,
        logger=None,
    ):

        self.logger = logger or get_logger(
            "PatchGroupSerializer",
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def save(
        self,
        patient_id: str,
        patch_groups: list,
        output_dir: str | Path,
    ) -> Path:
        """
        Save PatchGroups.

        Parameters
        ----------
        patient_id
            Patient identifier.

        patch_groups
            List of PatchGroup objects.

        output_dir
            Output directory.

        Returns
        -------
        Path
            Path to groups.json
        """

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        group_directory = output_dir / "group"
        group_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = group_directory / "groups.json"

        self.logger.info(
            "Saving patch groups : %s",
            output_file,
        )

        groups_dict = self._groups_to_dict(
            patient_id,
            patch_groups,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                groups_dict,
                fp,
                indent=4,
            )

        self.logger.info(
            "Patch groups saved successfully."
        )

        return output_file

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _groups_to_dict(
        self,
        patient_id: str,
        patch_groups: list,
    ) -> dict:

        return {

            "patient_id": patient_id,

            "number_of_groups": len(
                patch_groups,
            ),

            "groups": [

                {

                    "group_id": int(
                        group.group_id,
                    ),

                    "center_node": int(
                        group.center_node.node_id,
                    ),

                    "group_size": len(
                        group.member_nodes,
                    ),

                    "members": [

                        int(node.node_id)

                        for node

                        in group.member_nodes

                    ],

                }

                for group

                in patch_groups

            ],

        }