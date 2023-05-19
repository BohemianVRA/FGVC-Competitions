import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# custom script arguments
MODEL_ARCH = "vit_base_patch16_224"
NUM_CLASSES = 1000
CHECKPOINT_PATH = "SnakeCLEF2023-ViT_base_patch16_224-100E.pth"
WIDTH, HEIGHT = 224, 224


def run_inference(input_csv, output_csv, data_root_path):
    """Load model and dataloader and run inference."""
    # load input metadata with observation ids and image paths
    metadata_df = pd.read_csv(input_csv)
    assert "observation_id" in metadata_df
    assert "image_path" in metadata_df

    # load model with fine-tuned checkpoint
    print("Loading model with the fine-tuned checkpoint.")
    model = timm.create_model(
        MODEL_ARCH,
        pretrained=False,
        num_classes=NUM_CLASSES,  # remove classifier nn.Linear
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"), strict=True)
    model_mean = model.default_cfg["mean"]
    model_std = model.default_cfg["std"]

    # create dataloaders
    print("Creating DataLoader.")
    testset = ImageDataset(metadata_df, model_mean, model_std, WIDTH, HEIGHT, data_root_path)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # run inference
    print("Running inference.")
    preds = predict(model, testloader)

    # save predictions
    print(f"Saving predictions to '{output_csv}'.")
    user_pred_df = metadata_df[["observation_id"]].copy()
    user_pred_df["class_id"] = preds.argmax(1)
    # (dummy example) convert instance-based prediction into observation-based predictions
    # by keeping predictions of first instances in the dataframe
    user_pred_df = user_pred_df.drop_duplicates("observation_id", keep="first")
    user_pred_df.to_csv(output_csv)


class ImageDataset(Dataset):
    def __init__(self, df, model_mean, model_std, width, height, data_root_path):
        self.df = df
        self.transform = A.Compose(
            [
                A.Resize(width, height),
                A.Normalize(mean=model_mean, std=model_std),
                ToTensorV2(),
            ]
        )
        self.data_root_path = data_root_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = os.path.join(
            self.data_root_path,
            self.df["image_path"].iloc[idx],
        )
        assert os.path.isfile(file_path)
        # label = self.df['class_id'].iloc[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image


@torch.no_grad()
def predict(model, testloader):
    """Iterate through test dataloader and run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    preds_all = []
    for imgs in tqdm(testloader, total=len(testloader)):
        imgs = imgs.to(device)
        preds = model(imgs)
        preds_all.append(preds.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    return preds_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        help="Path to a file with observation ids and image paths.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-root-path",
        help="Path to a directory where images are stored.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-file",
        help="Path to a file where predict script will store predictions.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    output_csv = os.path.basename(args.output_file)
    if not output_csv.endswith(".csv"):
        output_csv = output_csv + ".csv"
    run_inference(
        input_csv=args.input_file,
        output_csv=output_csv,
        data_root_path=args.data_root_path,
    )
