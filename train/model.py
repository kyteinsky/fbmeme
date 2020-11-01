import torch
import torchvision.transforms as T
from concat import CatsAndDogs
from dataset import HM_dataset

import os
from tqdm import tqdm
import numpy as np
# from sklearn.metrics import accuracy_score
# from image_proc import im_encoded
from PIL import Image
import pickle
from tqdm import tqdm
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
wandb.init(project="fbmeme")

def accuracy(true, pred):
    assert len(true) == len(pred)
    acc = 0
    for t, p in zip(true, pred):
        t, p = t>=0.5, p>=0.5
        acc += 1 if t==p else 0
    return acc/len(true)


class Model():
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.imfc = im_encoded()

        self.concat_model = CatsAndDogs(
            dropout_p = self.hparams.get('dropout', 0.2),
            hidden_dim = self.hparams.get('hidden_dim', 4096)
        )

        self.output_path = os.path.join(self.hparams.get("output_path", "model-outputs"), '')
        if not os.path.isdir(self.output_path): os.mkdir(self.output_path)

        self.train_dataset = self._build_dataset("train_path")
        self.dev_dataset = self._build_dataset("dev_path")

        self.list_acc = []

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        self.model.to(self.device)
        self.model.eval()


    def get_features(self, imgs):
        save_output = []

        for name, layer in self.model.named_modules():
            if name == "transformer.encoder":
                handle = layer.register_forward_hook(
                    lambda self, input, output: save_output.append(output)
                )
        
        with torch.no_grad():
            out = self.model(imgs)
            save_output = save_output[0].detach()
            handle.remove()
        
        return save_output


    def train_step(self, batch):
        self.concat_model.train()

        preds = self.concat_model(
            ftext=batch["text"],
            image=batch["image"],
            face_lm=batch["face_lm"],
            label=batch["label"]
        )

        # print('train label:',batch['label'].detach().cpu().numpy().tolist())
        # print('train pred:', preds.detach().cpu().numpy().tolist())

        acc = accuracy(batch['label'].detach().cpu().numpy().tolist(), preds.detach().cpu().numpy().tolist())
        # return {'loss': loss, 'acc': acc}
        return preds, acc

    @torch.no_grad()
    def val_step(self, batch):
        preds = self.concat_model(
            ftext=batch["text"],
            image=batch["image"],
            face_lm=batch["face_lm"],
            label=batch["label"]
        )

        # print('val label:',batch['label'])
        # print('val pred:', preds.detach().cpu().numpy().tolist())

        acc = accuracy(batch['label'].detach().cpu().numpy().tolist(), preds.detach().cpu().numpy().tolist())
        # return {'loss': loss, 'acc': acc}
        return preds, acc

    def img_shape_cor(self, image):
        image = image[:, :1536, :]
        return torch.cat([
            image,
            torch.zeros(image.size()[0], 1536-image.size()[1], 256).type_as(image) # padding
        ], dim=1).reshape(image.size()[0], 192, 2048)
    
    def ftext_features(self, _id):
        with open(os.path.join('features/text/', str(_id.item())), 'rb') as f:
            ftext = pickle.load(f)
        ftext = ftext.detach()

        ftext = torch.cat([
            ftext, # (1, x, 1024)
            torch.zeros(1, 100-ftext.size()[1], 1024)
        ], dim=1)
        
        ftext = ftext.reshape(ftext.size()[0], -1, 2048)

        return ftext
        

    def face_lm_features(self, _id):
        with open(os.path.join('features/face_lm/', str(_id.item())), 'rb') as f:
            face_lm = pickle.load(f)
        face_lm = torch.tensor(face_lm, dtype=torch.float).unsqueeze(0).detach()

        return face_lm


    def train(self):
        # wandb 
        wandb.watch(self.concat_model)

        # data loaders
        train_loader = self.get_train_dataloader()
        val_loader = self.get_val_dataloader()

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.concat_model.parameters(), 
            lr=self.hparams.get("lr", 0.001)
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)


        # load from ckpt
        if os.path.isfile(os.path.join(self.output_path, 'model.pt')):
            self.concat_model = torch.load(os.path.join(self.output_path, 'model.pt'))
            print('Checkpoint loaded!')
        
        self.concat_model.to(self.device)


        # main loop
        print('Start training!')

        for epoch in range(self.hparams.get('max_epochs', 10)):
            print('EPOCH: ',epoch+1)
            running_loss = 0
            running_acc = 0
            pbar = tqdm(total = int(len(self.train_dataset)/self.hparams['batch_size']))

            for batch_idx, batch in enumerate(train_loader):
                pbar.update(1)

                self.concat_model.train()
                optimizer.zero_grad()

                image_features = torch.tensor([])
                text = torch.tensor([])
                face = torch.tensor([])

                # ftext and face_lm features
                for _id in batch['id']:
                    ftext = self.ftext_features(_id)
                    face_lm = self.face_lm_features(_id)
                    text = torch.cat([text, ftext])
                    face = torch.cat([face, face_lm])

                tmp_imgs = []
                # get image features
                for img in batch['image']:
                    img = Image.open(img).convert('RGB')
                    img_transform = self.get_train_transforms()
                    img = img_transform(img).to(self.device) #.unsqueeze(0)
                    tmp_imgs.append(img)
                
                # print('imgs:',tmp_imgs)

                imgs = self.get_features(tmp_imgs).permute(1, 0, 2)
                del tmp_imgs
                imgs = self.img_shape_cor(imgs)

                # (x, 1, 256) ->  (1, x, 256)
                # image_features = torch.cat([
                #     image_features,
                #     self.img_shape_cor(img)
                #     # (1, x, 256) -> (1, 1536, 256) -> (1, 192, 2048)
                # ])

                # data to device
                batch['text'] = text.to(self.device)
                batch['image'] = imgs.to(self.device)
                batch['face_lm'] = face.to(self.device)
                batch['label'] = batch['label'].to(self.device)

                # print(f'Batch: {batch_idx+1} out of {int(len(self.train_dataset)/self.hparams["batch_size"])+1} == progress: {round((batch_idx/len(self.train_dataset))*100, 1)}%')

                preds, acc = self.train_step(batch)
                loss = criterion(preds, batch['label'])

                loss.backward()
                optimizer.step()
                running_loss = running_loss + loss.item()
                running_acc = running_acc + acc

                gc.collect()

                if (batch_idx+1)*self.hparams['batch_size'] % self.hparams.get('save_every', 200) == 0:
                    # saving ckpt -- best
                    self.save_ckpt(running_acc/self.hparams.get('save_every', 200))


                if (batch_idx+1)*self.hparams['batch_size'] % self.hparams.get('loss_every', 50) == 0:
                    
                    print('------------------------------------------')
                    print(f'EPOCH: {epoch+1}/{self.hparams["max_epochs"]}')
                    print(f'Loss: {running_loss/self.hparams.get("loss_every", 50)}')
                    print(f'Accuracy: {running_acc/self.hparams.get("loss_every", 50)}')
                    wandb.log({"Train Accuracy": (running_acc/self.hparams.get("loss_every", 50)), "Train Loss": (running_loss/self.hparams.get("loss_every", 50))})
                    running_loss = 0
                    running_acc = 0




            # validation
            with torch.no_grad():
                running_val_loss, running_val_acc = 0, 0
                print('Validating..')
                for _,val_batch in enumerate(val_loader):

                    image_features = torch.tensor([])
                    ftext = torch.tensor([])
                    face = torch.tensor([])

                    # ftext and face_lm features
                    for _id in batch['id']:
                        ftext = self.ftext_features(_id)
                        face_lm = self.face_lm_features(_id)
                        text = torch.cat([text, ftext])
                        face = torch.cat([face, face_lm])

                    tmp_imgs = []
                    # get image features
                    for img in val_batch['image']:
                        img = Image.open(img).convert('RGB')
                        img_transform = self.get_val_transforms()
                        img = img_transform(img).to(self.device) #.unsqueeze(0)
                        tmp_imgs.append(img)
                    
                    # print('imgs:',tmp_imgs)

                    imgs = self.get_features(tmp_imgs).permute(1, 0, 2)
                    del tmp_imgs
                    imgs = self.img_shape_cor(imgs)

                    # data to device
                    val_batch['text'] = text.to(self.device)
                    val_batch['image'] = imgs.to(self.device)
                    val_batch['face_lm'] = face.to(self.device)
                    val_batch['label'] = val_batch['label'].to(self.device)

                    print(f'Val Batch: {_+1} out of {int(len(self.dev_dataset)/self.hparams["batch_size"])+1}')

                    val_preds, val_acc = self.val_step(val_batch)
                    val_loss = criterion(val_preds, val_batch['label'])

                    running_val_loss = running_val_loss + val_loss.item()
                    running_val_acc = running_val_acc + val_acc
                
                print(f'Val Loss: {running_val_loss/len(self.dev_dataset)}')
                print(f'Val Accuracy: {running_val_acc/len(self.dev_dataset)}')
                wandb.log({"Val Accuracy": (running_val_acc/len(self.dev_dataset)), "Val Loss": (losrunning_val_loss/len(self.dev_dataset))})
            
                scheduler.step(running_val_acc/len(self.dev_dataset))


        print('Finished training!')
        print('Train Accuracy:',running_acc/self.hparams.get("loss_every", 50))
        print('Train Loss:',running_loss/self.hparams.get("loss_every", 50))
        print('Val Accuracy:',running_val_acc/len(self.dev_dataset))
        print('Val Loss:',running_val_loss/len(self.dev_dataset))


    def save_ckpt(self, acc):
        if (self.list_acc != [] and all(acc>i for i in self.list_acc)) or self.list_acc == []:
            try:
                torch.save(self.concat_model, os.path.join(self.output_path, 'model.pt'))
                with open(os.path.join(self.output_path, f'lr-{self.hparams["lr"]}-acc-{acc}'), 'w') as f:
                    f.write('model.pt')
                    f.close()
                # torch.save(self.concat_model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
            except Exception as e:
                print("Exception:",e)
                with open(os.path.join(self.output_path, f'lr-{self.hparams["lr"]}-acc-{acc}'), 'w') as f:
                    f.write('model.pt')
                    f.close()
                with open(os.path.join(self.output_path, 'model.pt'), 'wb') as f:
                    pickle.dump(self.concat_model, f, pickle.HIGHEST_PROTOCOL)
            print('Saved checkpoint!')


    def get_train_transforms(self):
        return T.Compose(
            [
                T.Resize(800),
                T.RandomHorizontalFlip(self.hparams['augmentation']['h_flip']),
                T.RandomRotation(self.hparams['augmentation']['rotation']),
                # T.ColorJitter(**self.hparams['augmentation']['color_jit']),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    def get_val_transforms(self):
        return T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    
    def _build_dataset(self, name):
        return HM_dataset(
            data_path = self.hparams.get('data_path'),
            jsonl_path = self.hparams.get(name),
            # image_transform = self.get_train_transforms() if 'train' in name else self.get_val_transforms(),
            balance = self.hparams.get('balance', False)
        )
    

    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            shuffle=True, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16)
        )

    def get_val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset, 
            shuffle=False, 
            batch_size=self.hparams.get("batch_size", 4), 
            num_workers=self.hparams.get("num_workers", 16)
        )
    
