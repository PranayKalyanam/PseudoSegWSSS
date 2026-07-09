from __future__ import annotations

from dataclasses import dataclass, field

from data.patch.patch import Patch


@dataclass(slots=True)
class PatchDataset:
    """
    Represents all patches extracted from a single patient.

    This object is produced by the PatchLoader and
    encapsulates every patch generated during the
    patch extraction stage.
    """

    # --------------------------------------------------
    # Extracted Patches
    # --------------------------------------------------

    patches: list[Patch] = field(
        default_factory=list
    )

    # --------------------------------------------------
    # Convenience Properties
    # --------------------------------------------------

    @property
    def number_of_patches(self) -> int:
        """
        Total extracted patches.
        """
        return len(self.patches)

    @property
    def is_empty(self) -> bool:
        """
        Returns True if no patches exist.
        """
        return self.number_of_patches == 0
    
    @property
    def has_patches(self) -> bool:
        """
        Returns True if patches exist.
        """
        return self.number_of_patches > 0
