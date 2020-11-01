
import torch
import pandas as pd
import pickle
import os
# from PIL import Image


class HM_dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path,
        jsonl_path,
        # image_transform,
        balance=False,
        random_state=1012,
    ):

        self.data_path = data_path
        self.samples_frame = pd.read_json(os.path.join(self.data_path, jsonl_path), lines=True)
        # self.image_transform = image_transform
        
        if balance:
            neg = self.samples_frame[self.samples_frame.label.eq(0)]
            pos = self.samples_frame[self.samples_frame.label.eq(1)]
            self.samples_frame = pd.concat([
                neg.sample(pos.shape[0], random_state=random_state), 
                pos
            ])

        self.samples_frame = self.samples_frame.reset_index(drop=True)
        self.samples_frame['img'] = data_path + self.samples_frame['img'].astype(str)

        # https://github.com/drivendataorg/pandas-path
        # if not self.samples_frame.img.path.exists().all():
        #     raise FileNotFoundError
        # if not self.samples_frame.img.path.is_file().all():
        #     raise TypeError


    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]

        # image = Image.open(self.samples_frame.loc[idx, "img"]).convert("RGB")
        # if self.image_transform: image = self.image_transform(image)
        image = self.samples_frame.loc[idx, "img"]

        # text = self.samples_frame.loc[idx, "text"] ### for kg
      

        # ftext
        # face_lm

        if "label" in self.samples_frame.columns:
            label = torch.Tensor(
                [self.samples_frame.loc[idx, "label"]]
            ).squeeze()
            sample = {
                "id": img_id, 
                "image": image,
                # "face_lm": face_lm,
                # "text": ftext,
                "label": label
            }
        else:
            sample = {
                "id": img_id, 
                "image": image, 
                # "face_lm": face_lm,
                # "text": ftext
            }

        # id
        # image is a path
        # face_lm is a feature tensor (3, 136)
        # text is a feature tensor (x, 1024)
        # label (0, 1)

        return sample

