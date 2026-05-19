import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

from utils import find_active_player, find_cards_per_player, find_central_card

def main():

    project_root = Path("./")
    test_images_dir = project_root / "test_images"

    result = pd.DataFrame(columns=["image_id","center_card","active_player","player_1_cards","player_2_cards","player_3_cards","player_4_cards"])

    for path in list(test_images_dir.glob("*.jpg")):
        original_img = np.array(Image.open(path))
        row = []
        image_id = str(path).split("\\")[1].split(".")[0]
        center_card = find_central_card(original_img)
        active_player = find_active_player(original_img)
        player_cards = find_cards_per_player(original_img)
        player_1_cards = str(player_cards[1]).replace(",", ";")
        player_2_cards = str(player_cards[2]).replace(",", ";")
        player_3_cards = str(player_cards[3]).replace(",", ";")
        player_4_cards = str(player_cards[4]).replace(",", ";")

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