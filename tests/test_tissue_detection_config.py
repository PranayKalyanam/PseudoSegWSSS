import sys
import types
from argparse import Namespace

cv2_stub = types.ModuleType("cv2")
cv2_stub.threshold = lambda *args, **kwargs: (None, None)
cv2_stub.GaussianBlur = lambda *args, **kwargs: None
cv2_stub.morphologyEx = lambda *args, **kwargs: None
cv2_stub.connectedComponentsWithStats = lambda *args, **kwargs: (None, None, None, None)
cv2_stub.findContours = lambda *args, **kwargs: []
cv2_stub.boundingRect = lambda *args, **kwargs: (0, 0, 0, 0)
cv2_stub.rectangle = lambda *args, **kwargs: None
cv2_stub.cvtColor = lambda *args, **kwargs: None
cv2_stub.resize = lambda *args, **kwargs: None
cv2_stub.imwrite = lambda *args, **kwargs: True

sys.modules.setdefault("cv2", cv2_stub)

from preprocessing.tissue_detection import BaseTissueDetector


class DummyDetector(BaseTissueDetector):
    def detect(self, image):
        return None


def test_detector_accepts_namespace_config():
    config = Namespace(
        tissue_threshold=0.8,
        minimum_region_area=123,
        gaussian_kernel=(3, 3),
        morphology_kernel_size=7,
        open_iterations=1,
        close_iterations=2,
        debug=True,
        save_intermediate_results=False,
        output_directory=None,
    )

    detector = DummyDetector(config)

    assert detector.config.tissue_threshold == 0.8
    assert detector.config.minimum_region_area == 123
    assert detector.config.gaussian_kernel == (3, 3)
    assert detector.config.morphology_kernel_size == 7
    assert detector.config.open_iterations == 1
    assert detector.config.close_iterations == 2
