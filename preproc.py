import os
import torch
import pandas as pd
# from PIL import Image
from pathlib import Path
import pickle
from tqdm import tqdm
import gc
import sys

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

        self.train_ds = HM_dataset(
            data_dir/'annotations/train.jsonl',
            data_dir
        )
        self.test_ds = HM_dataset(
            data_dir/'annotations/test_seen.jsonl',
            data_dir
        )
        self.dev_ds = HM_dataset(
            data_dir/'annotations/dev_seen.jsonl',
            data_dir
        )

    def save(self, ffolder='features/'):
        # mkdir batch_size in `ffolder`
        # mkdir face/ text/ in batch_size/

        if not os.path.isdir(ffolder): os.mkdir(ffolder)

        path_var = dict()

        for i in ['face', 'text']:
            tmp_path_var = os.path.join(ffolder, i)
            if not os.path.isdir(tmp_path_var):
                os.makedirs(tmp_path_var)
            path_var[i] = tmp_path_var


        for _,obj in enumerate([self.train_ds, self.dev_ds, self.test_ds]):
            for dt in obj:
                # for face
                fc = self.face_obj.dine(str(dt['image']))
                with open(str(Path(path_var['face'])) +'/'+ str(dt['id']), 'wb') as f:
                    pickle.dump(fc, f, pickle.HIGHEST_PROTOCOL)

                del fc
                del f

                # for text
                fc = self.rob.get_features(dt['text'])
                with open(str(Path(path_var['text'])) +'/'+ str(dt['id']), 'wb') as f:
                    pickle.dump(fc, f, pickle.HIGHEST_PROTOCOL)
            
            # progress(_, len(obj), 'dataset no. '+str(_))
            gc.collect()



# def progress(count, total, status=''):
#     bar_len = 60
#     filled_len = int(round(bar_len * count / float(total)))

#     percents = round(100.0 * count / float(total), 1)
#     bar = '=' * filled_len + '-' * (bar_len - filled_len)

#     sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
#     sys.stdout.flush()


if __name__ == '__main__':
    features = save_features()
    features.save()
