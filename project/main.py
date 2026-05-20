import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path

from utils import classify_folder, FourierDiscriminator, save_result

def main():

    project_root = Path("./")
    train_images_dir = project_root / "train_images"
    test_images_dir = project_root / "test_images"

    classifier = FourierDiscriminator(1, 9.58)

    classifier.fit()

    result = classify_folder(test_images_dir, classifier)

    save_result(result)

if __name__ == "__main__":
    main()