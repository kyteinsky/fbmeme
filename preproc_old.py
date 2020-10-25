import os
import torch
import pandas as pd
# from PIL import Image
from pathlib import Path
import pickle

from face_dinner import face_dinner
from robert import roberta_enc

class HM_dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path,
        img_dir,
        balance=False,
        random_state=1012,
    ):

        self.samples_frame = pd.read_json(data_path, lines=True)
        
        if balance:
            neg = self.samples_frame[self.samples_frame.label.eq(0)]
            pos = self.samples_frame[self.samples_frame.label.eq(1)]
            self.samples_frame = pd.concat([
                neg.sample(pos.shape[0], random_state=random_state), 
                pos
            ])

        self.samples_frame = self.samples_frame.reset_index(drop=True)
        self.samples_frame.img = self.samples_frame.apply(
            lambda row: (img_dir / row.img), axis=1)

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
        image = self.samples_frame.loc[idx, "img"]

        text = self.samples_frame.loc[idx, "text"]

        if "label" in self.samples_frame.columns:
            label = torch.Tensor(
                [self.samples_frame.loc[idx, "label"]]
            ).long().squeeze()
            sample = {
                "id": img_id, 
                "image": image, 
                "text": text, 
                "label": label
            }
        else:
            sample = {
                "id": img_id, 
                "image": image, 
                "text": text
            }

        return sample

class save_features():
    def __init__(self, data_folder='data'):

        self.face_obj = face_dinner()
        self.rob = roberta_enc()

        data_dir = Path.cwd() / data_folder

        train_ds = HM_dataset(
            data_dir/'annotations/train.jsonl',
            data_dir
        )
        test_ds = HM_dataset(
            data_dir/'annotations/test_seen.jsonl',
            data_dir
        )
        dev_ds = HM_dataset(
            data_dir/'annotations/dev_seen.jsonl',
            data_dir
        )
        # self.params = {'batch_size': 64,
        #     'shuffle': True,
        #     'num_workers': 6}

        # self.train_gen = torch.utils.data.DataLoader(train_ds, **self.params)
        # self.dev_gen = torch.utils.data.DataLoader(dev_ds, **self.params)
        # self.test_gen = torch.utils.data.DataLoader(test_ds, **self.params)

    def save(self, ffolder='features/'):
        # mkdir batch_size in `ffolder`
        # mkdir face/ text/ in batch_size/

        if not os.path.isdir(ffolder): os.mkdir(ffolder)

        path_var = []

        for i in ['face', 'text']:
            tmp_path_var = os.path.join(self.params['batch_size'], i)
            if os.path.isdir(tmp_path_var):
                raise f'{tmp_path_var} => path already exists!'
            os.makedirs(tmp_path_var)
            path_var.append(tmp_path_var)
        
        for obj in [self.train_gen, self.dev_gen, self.test_gen]:
            for sample in obj:
                fc = self.face_obj.dine(sample)


if __name__ == '__main__':
    features = save_features()
    features.save()
