from types import SimpleNamespace

import numpy as np

import preprocess


class DummyImageLoader:
    def __init__(self, config):
        self.config = config
        self.patient_ids = ["patient-1"]

    def load_patient(self, patient_id):
        self.loaded_patient = patient_id

    def get_image(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def get_mask(self):
        return np.zeros((32, 32), dtype=np.uint8)


class DummyTissueDetector:
    def __init__(self, config):
        self.config = config

    def detect(self, image, patient_id=""):
        return SimpleNamespace(regions=[])


class DummyPatchExtractor:
    def __init__(self, config):
        self.config = config


class DummyLabelGenerator:
    def __init__(self, config):
        self.config = config


class DummyDatasetWriter:
    def __init__(self, config):
        self.config = config

    def write_dataset(self, manifest):
        return SimpleNamespace(success=True, statistics=SimpleNamespace(generated_files=0, failed_files=0, total_patches=0, average_tissue_percentage=0.0))


def test_main_uses_image_loader_patient_api(monkeypatch):
    config = SimpleNamespace(
        dry_run=False,
        image_dir="datasets/BCSS/imgs",
        mask_dir="datasets/BCSS/masks",
        output_dir="outputs",
        patch_size=224,
        overlap=0.5,
        magnification=10,
        workers=1,
        log_level="INFO",
    )

    monkeypatch.setattr(preprocess, "get_config", lambda: config)
    monkeypatch.setattr(preprocess, "ImageLoader", DummyImageLoader)
    monkeypatch.setattr(preprocess, "OtsuTissueDetector", DummyTissueDetector)
    monkeypatch.setattr(preprocess, "BasePatchExtractor", DummyPatchExtractor)
    monkeypatch.setattr(preprocess, "BCSSLabelGenerator", DummyLabelGenerator)
    monkeypatch.setattr(preprocess, "DatasetWriter", DummyDatasetWriter)

    assert preprocess.main() == preprocess.EXIT_SUCCESS
