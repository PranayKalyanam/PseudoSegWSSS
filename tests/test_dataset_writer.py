import cv2
import numpy as np

from dataio.dataset_writer import DatasetManifest, DatasetWriter, DatasetWriterConfig
from dataset.data_structures import PatchRecord


def _build_writer_and_record(tmp_path, patient_id, filename, labels, x=0, y=0):
    writer = DatasetWriter(DatasetWriterConfig(output_dir=tmp_path))

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)

    image_path = writer.patches_dir / patient_id / filename
    mask_path = writer.masks_dir / patient_id / filename
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    record = PatchRecord(
        patch_id=f"{patient_id}-{filename}",
        patient_id=patient_id,
        region_id=1,
        filename=filename,
        image=image,
        mask=mask,
        x=x,
        y=y,
        width=256,
        height=256,
        tissue_percentage=0.8,
        labels=labels,
        metadata={},
    )

    return writer, record


def test_verify_dataset_allows_same_filename_across_patients(tmp_path):
    writer, record_a = _build_writer_and_record(
        tmp_path,
        patient_id="patientA",
        filename="shared.png",
        labels=[1, 0, 0, 0],
    )
    _, record_b = _build_writer_and_record(
        tmp_path,
        patient_id="patientB",
        filename="shared.png",
        labels=[0, 1, 0, 0],
    )

    manifest = DatasetManifest(records=[record_a, record_b], patches=[object(), object()])

    report = writer.verify_dataset(manifest)

    assert report["duplicate_filenames"] == []
    assert report["valid"] is True


def test_verify_dataset_collects_invalid_labels_without_aborting(tmp_path):
    writer, valid_record = _build_writer_and_record(
        tmp_path,
        patient_id="patientA",
        filename="valid.png",
        labels=[1, 0, 0, 0],
    )
    _, invalid_record = _build_writer_and_record(
        tmp_path,
        patient_id="patientA",
        filename="invalid.png",
        labels=[0, 2, 0, 0],
    )

    manifest = DatasetManifest(records=[valid_record, invalid_record], patches=[object(), object()])

    report = writer.verify_dataset(manifest)

    assert report["invalid_labels"] == ["invalid.png"]
    assert report["valid"] is False
