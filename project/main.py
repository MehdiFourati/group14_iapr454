import pandas as pd
import numpy as np

from pathlib import Path

from project.src.utils import classify_folder, FourierDiscriminator, save_result
from project.src.dataset_creator import make_reference_group, build_rotated_descriptor_dataset

def main():

    project_root = Path(__file__).resolve().parent
    train_images_dir = project_root / "train_images"
    test_images_dir = project_root / "test_images"
    references_dir = project_root / "reference_images"
    print(references_dir)
    image_groups = make_reference_group(references_dir)
    datasets = {}
    all_images = {}
    all_contours = {}

    for label, imgs in image_groups.items():

        rotated_imgs, contours, descriptors = build_rotated_descriptor_dataset(imgs, label, n_rotations=21, angle_step=9)

        all_images[label] = rotated_imgs
        all_contours[label] = contours

        datasets[label] = np.concatenate(descriptors, axis=0)

    classifier = FourierDiscriminator(1, 8.5) 

    classifier.fit(datasets)

    result = classify_folder(test_images_dir, classifier)

    save_result(result)

if __name__ == "__main__":
    main()