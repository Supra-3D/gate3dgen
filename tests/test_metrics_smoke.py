from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gate3dgen.evaluation.metrics2d import basic_metrics


def test_identical_images_have_perfect_basic_scores(tmp_path: Path) -> None:
    image = Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8), mode="RGB")
    prediction = tmp_path / "prediction.png"
    target = tmp_path / "target.png"
    image.save(prediction)
    image.save(target)

    metrics = basic_metrics(prediction, target)
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["ssim"] == 1.0

