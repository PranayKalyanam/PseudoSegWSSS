from data import patient
from methods.visualization.patch_visualizer import PatchVisualizer


PatchVisualizer.save(
            patient,
            f"outputs/patch_loader/{patient.patient_id}",
        )