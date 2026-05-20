import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path

from utils import find_active_player, find_cards_per_player, FourierDiscriminator

def main():

    project_root = Path("./")
    train_images_dir = project_root / "train_images"
    test_images_dir = project_root / "test_images"

    result = pd.DataFrame(columns=["image_id","center_card","active_player","player_1_cards","player_2_cards","player_3_cards","player_4_cards"])

    classifier = FourierDiscriminator(1, 9.58)

    classifier.fit()

    for path in list(test_images_dir.glob("*.jpg")):
        original_img = np.array(Image.open(path))
        row = []
        player_cards = find_cards_per_player(original_img, classifier)
        image_id = str(path).split("\\")[1].split(".")[0]
        center_card = str(player_cards["center"])[0]
        active_player = find_active_player(original_img)
        player_1_cards = str(player_cards["p1"]).replace(",", ";")
        player_2_cards = str(player_cards["p2"]).replace(",", ";")
        player_3_cards = str(player_cards["p3"]).replace(",", ";")
        player_4_cards = str(player_cards["p4"]).replace(",", ";")

        row.append(image_id)
        row.append(center_card)
        row.append(active_player)
        row.append(player_1_cards)
        row.append(player_2_cards)
        row.append(player_3_cards)
        row.append(player_4_cards)

        result.loc[len(result)] = row

    result.to_csv("submission.csv")

if __name__ == "__main__":
    main()