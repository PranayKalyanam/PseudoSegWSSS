from __future__ import annotations

from dataclasses import dataclass, field

from data.image.image_pair import ImagePair
from data.image.image_data import ImageData
from data.tissue.tissue_region import TissueRegion
from data.patch.patch import Patch
from data.graph.graph import Graph
from data.dataset.dataset_manifest import DatasetManifest
from data.tissue.tissue_mask import TissueMask


@dataclass(slots=True)
class Patient:
    """
    Represents one patient throughout the complete
    preprocessing pipeline.

    Every preprocessing stage enriches this object.

    Dataset Discovery
            ↓
    Image Loading
            ↓
    Tissue Detection
            ↓
    Patch Extraction
            ↓
    Weak Label Generation
            ↓
    Graph Construction
            ↓
    Dataset Generation
    """

    # --------------------------------------------------
    # Patient Identity
    # --------------------------------------------------

    image_pair: ImagePair

    # --------------------------------------------------
    # Image Loading
    # --------------------------------------------------

    image_data: ImageData | None = None

    # --------------------------------------------------
    # Tissue Detection
    # --------------------------------------------------



    tissue_mask: TissueMask | None = None

    # --------------------------------------------------
    # Patch Extraction
    # --------------------------------------------------

    patches: list[Patch] = field(
        default_factory=list
    )

    # --------------------------------------------------
    # Graph Construction
    # --------------------------------------------------

    graph: Graph | None = None

    # --------------------------------------------------
    # Dataset Generation
    # --------------------------------------------------

    dataset_manifest: DatasetManifest | None = None
    
    # --------------------------------------------------
    # Convenience Properties
    # --------------------------------------------------

    @property
    def patient_id(self) -> str:
        """
        Returns the patient identifier.
        """
        return self.image_pair.patient_id
   
    @property
    def image_path(self):
        """
        Returns the image path.
        """
        return self.image_pair.image_path

    @property
    def mask_path(self):
        """
        Returns the annotation mask path.
        """
        return self.image_pair.mask_path

    @property
    def image_filename(self) -> str:
        """
        Returns the image filename.
        """
        return self.image_pair.image_filename

    @property
    def mask_filename(self) -> str:
        """
        Returns the mask filename.
        """
        return self.image_pair.mask_filename

    @property
    def has_image(self) -> bool:
        return self.image_data is not None

    @property
    def has_tissue_regions(self) -> bool:
        return len(self.tissue_regions) > 0

    @property
    def has_patches(self) -> bool:
        return len(self.patches) > 0

    @property
    def has_graph(self) -> bool:
        return self.graph is not None

    @property
    def is_processed(self) -> bool:
        """
        Returns True when preprocessing is complete.
        """
        return self.dataset_manifest is not None

    @property
    def image(self):
        if self.image_data is None:
            return None
        return self.image_data.image

    @property
    def mask(self):
        if self.image_data is None:
            return None
        return self.image_data.mask

    @property
    def metadata(self):
        if self.image_data is None:
            return None
        return self.image_data.metadata
    @property
    def has_tissue(self) -> bool:
        """
        Returns True if tissue detection has been completed.
        """
        return self.tissue_mask is not None


    @property
    def tissue_regions(self):
        """
        Returns all detected tissue regions.
        """
        if self.tissue_mask is None:
            return []

        return self.tissue_mask.regions


    @property
    def tissue_statistics(self):
        """
        Returns tissue statistics.
        """
        if self.tissue_mask is None:
            return None

        return self.tissue_mask.statistics


    @property
    def tissue_binary_mask(self):
        """
        Returns the binary tissue mask.
        """
        if self.tissue_mask is None:
            return None

        return self.tissue_mask.mask


    @property
    def number_of_tissue_regions(self) -> int:
        """
        Returns the number of detected tissue regions.
        """
        if self.tissue_mask is None:
            return 0

        return self.tissue_mask.number_of_regions
    
    
    @property
    def working_image(self):
        if self.image_data is None:
            return None
        return self.image_data.working_image


    @property
    def working_mask(self):
        if self.image_data is None:
            return None
        return self.image_data.working_mask