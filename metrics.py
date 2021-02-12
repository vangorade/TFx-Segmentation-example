from objects import *
from functional import *
class IOUScore(Metric):
    def __init__(self, class_weights=None, threshold=None, smooth=1e-5, name=None):
        name = name or "iou_score"
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1.
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return iou_score(
            y_true,
            y_pred,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=self.threshold
        )
