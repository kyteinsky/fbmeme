
import torch
import torchvision.transforms as T
from PIL import Image


class SaveOutput:
    def __init__(self):
        self.outputs = dict()
        
    def __call__(self, module, module_in, module_out):
        self.outputs['encoder'] = module_out

    def clear(self):
        self.outputs = dict()


class im_encoded():
    def __init__(self):
        """
        -> Load model once and for all
        -> Create a hook for saving features from encoder (save_output)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        # self.deivce = xm.xla_device() if xm.xla_device().index != None else device

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.save_output = SaveOutput()

        for name, layer in self.model.named_modules():
            if name == "transformer.encoder":
                handle = layer.register_forward_hook(self.save_output)

    def get_features(self, imgs):
        # transform = T.Compose([
        #     T.Resize(800),
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        # img = Image.open(img_path)
        # img = transform(img).unsqueeze(0)
        
        # imgs = imgs.to(self.device)

        out = self.model(imgs)
        output = self.save_output.outputs['encoder'].detach()

        del out
        self.save_output = None
        
        return output

