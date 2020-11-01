import torch
from transformer_encoder import Transformer


class CatsAndDogs(torch.nn.Module):
    def __init__(
        self,
        dropout_p,
        hidden_dim=4096
    ):
        super(CatsAndDogs, self).__init__()

        self.transformer = Transformer(dropout=dropout_p)
        # self.zero1 = torch.zeros((1, 1024))
        # self.zero2 = torch.zeros((1, 256))
        self.zero3 = torch.zeros((1, 2048))

        self.dropout = torch.nn.Dropout(dropout_p)
        self.lin_face_lm = torch.nn.Linear(5*136, 2048)
        self.layer_norm = torch.nn.BatchNorm1d(2048) # LayerNorm
        self.relu = torch.nn.ReLU()

        self.lin1 = torch.nn.Linear((50+1+192)*2048, hidden_dim)
        self.norm1 = torch.nn.BatchNorm1d(hidden_dim) # LayerNorm
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, 1)


    def forward(self, ftext, image, face_lm, label=None):

        # if ftext.size()[1] % 2 != 0: # ODD FIX
        #     ftext = torch.cat((
        #         ftext,
        #         self.zero1.unsqueeze(0).repeat(ftext.size()[0], 1, 1).type_as(ftext)
        #     ), dim=1)

        image_features = image.type_as(image) # (N, x, 256)

        # if image_features.size()[1] % 8 != 0: # EIGHT FIX
        #     n = image_features.size()[1]
        #     div = [(n+i) for i in range(8) if (n+i)%8==0]
        #     image_features = torch.cat((
        #         image_features,
        #         self.zero2.unsqueeze(0).repeat(image_features.size()[0], div[0]-n, 1).type_as(image_features)
        #     ), dim=1)

        # image_features = image_features.reshape(image_features.size()[0], -1, 2048)
        # image_features = image_features[:, :192, :]

        face_lm = face_lm.reshape(-1, 5*136)
        face_lm.requires_grad = True
        face_lm = self.relu(self.lin_face_lm(face_lm))
        face_lm = face_lm.unsqueeze(1)

        x = torch.cat([
            ftext,
            image_features,
            face_lm
        ], dim=1)
        
        del image_features
        del face_lm
        del ftext

        x = self.transformer(x)

        if (1+50+192-x.size()[1]) != 0:
            x = torch.cat([
                x,
                # torch.zeros((x.size()[0], (1+50+192-x.size()[1]), 2048), device=device)
                self.zero3.unsqueeze(0).repeat(x.size()[0], (1+50+192-x.size()[1]), 1).type_as(x)
            ], dim=1)

        # r = self.zero3.unsqueeze(0).repeat(x.size()[0], 1, 1) # big zero tensor 50+1+192
        # rs = r.size()
        # r[:rs[0],:rs[1],:rs[2]] = x # replace values

        # linear head -- max = 50 + 1 + 192, 2048
        x = x.reshape(-1, (50+1+192)*2048)

        x = self.norm1(self.relu(self.lin1(x)))
        x = self.dropout(x)
        x = self.norm1(self.relu(self.lin2(x)))

        x = self.lin3(x)
        x = x.squeeze()
        
        label = label.float().type_as(x)
        # if label.size()[0] == 1: label = label.unsqueeze(0)

        

        # print(torch.cuda.memory_stats(device))
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()

        return x

