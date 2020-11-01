from model import HatefulMemesModel

data_path = '../data/'

hparams = {
    
    "data_path": data_path,
    "train_path": 'annotations/train.jsonl',
    "dev_path": 'annotations/dev_seen.jsonl',
    "output_path": "model-outputs",

    "balance": False,
    "hidden_dim": 4096,
    "lr": 0.0001,
    "max_epochs": 10,
    "n_gpu": 1,
    "batch_size": 4,

    # allows us to "simulate" having larger batches 
    "accumulate_grad_batches": 16,
    "early_stop_patience": 3,
    "gradient_clip_value": 0.99,

    "augmentation":{
        "h_flip": 0.2,
        "rotation": 20,
        "color_jit":{
            "brightness":0.2,
            "contrast":0.2,
            "saturation":0.1,
            "hue":0.15
        }
    }
}

hateful_memes_model = HatefulMemesModel(hparams=hparams)
hateful_memes_model.fit()

# auto-lr
