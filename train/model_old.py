import pytorch_lightning as pl
import torchvision.transforms as T
from concat import CatsAndDogs
from dataset import HM_dataset
import torch


# for the purposes of this post, we'll filter
# much of the lovely logging info from our LightningModule
# warnings.filterwarnings("ignore")
# logging.getLogger().setLevel(logging.WARNING)


class HatefulMemesModel(pl.LightningModule):
	def __init__(self, hparams):
		for data_key in ["train_path", "dev_path", "data_path",]:
			# ok, there's one for-loop but it doesn't count
			if data_key not in hparams.keys():
				raise KeyError(
					f"{data_key} is a required hparam in this model"
				)

		super(HatefulMemesModel, self).__init__()
		self.hparams = hparams
		
		# assign some hparams that get used in multiple places
		self.output_path = Path(
			self.hparams.get("output_path", "model-outputs")
		)
		self.output_path.mkdir(exist_ok=True)
		
		# instantiate transforms, datasets
		self.image_transform = self._build_image_transform()
		self.train_dataset = self._build_dataset("train_path")
		self.dev_dataset = self._build_dataset("dev_path")
		
		# set up model and training
		self.train_acc = pl.metrics.Accuracy()
    	self.valid_acc = pl.metrics.Accuracy()

		self.model = self._build_model()
		self.trainer_params = self._get_trainer_params()
	
	## Required LightningModule Methods (when validating) ##
	
	def forward(self, text, image, label=None):
		return self.model(text, image, face_lm, label)

	def training_step(self, batch, batch_nb):
		preds, loss = self.forward(
			text=batch["text"],
			image=batch["image"],
			face_lm=batch["face_lm"],
			label=batch["label"]
		)
		self.train_acc(preds, batch["label"])
		
		return {"loss": loss, "train_acc": self.train_acc}

	def validation_step(self, batch, batch_nb):
		preds, loss = self.eval().forward(
			text=batch["text"],
			image=batch["image"],
			face_lm=batch["face_lm"],
			label=batch["label"]
		)
		self.valid_acc(preds, batch["label"])

		return {"batch_val_loss": loss, "valid_acc": self.valid_acc}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack(
			tuple(
				output["batch_val_loss"] 
				for output in outputs
			)
		).mean()
		
		return {
			"val_loss": avg_loss,
			"progress_bar":{"avg_val_loss": avg_loss}
		}

	def configure_optimizers(self):
		optimizers = [
			torch.optim.AdamW(
				self.model.parameters(), 
				lr=self.hparams.get("lr", 0.001)
			)
		]
		schedulers = [
			torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizers[0]
			)
		]
		return optimizers, schedulers
	
	@pl.data_loader
	def train_dataloader(self):
		return torch.utils.data.DataLoader(
			self.train_dataset, 
			shuffle=True, 
			batch_size=self.hparams.get("batch_size", 4), 
			num_workers=self.hparams.get("num_workers", 16)
		)

	@pl.data_loader
	def val_dataloader(self):
		return torch.utils.data.DataLoader(
			self.dev_dataset, 
			shuffle=False, 
			batch_size=self.hparams.get("batch_size", 4), 
			num_workers=self.hparams.get("num_workers", 16)
		)
	
	## Convenience Methods ##
	
	def fit(self):
		self._set_seed(self.hparams.get("random_state", 1242))
		self.trainer = pl.Trainer(**self.trainer_params)
		self.trainer.fit(self)
		
	def _set_seed(self, seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	
	def _build_image_transform(self):
		image_transform = T.Compose(
			[
				T.Resize(800),
				T.RandomHorizontalFlip(self.hparams['augmentation']['h_flip']),
				T.RandomRotation(self.hparams['augmentation']['rotation']),
				T.ColorJitter(**self.hparams['augmentation']['color_jit']),
				T.ToTensor(),
				T.Normalize(
					mean=(0.485, 0.456, 0.406), 
					std=(0.229, 0.224, 0.225)
				),
			]
		)
		return image_transform

	def _build_dataset(self, dataset_key):
		return HM_dataset(
			data_path = self.hparams.get('data_path'),
			jsonl_path = self.hparams.get(dataset_key),
			image_transform = self.image_transform,
			balance = self.hparams.get('balance', False)
		)

	def _build_model(self):

		return CatsAndDogs(
			loss_fn=torch.nn.CrossEntropyLoss(),
			dropout_p=self.hparams.get("dropout_p", 0.2),
			hidden_dim=self.hparams['hidden_dim']
		)
	
	
	def _get_trainer_params(self):
		checkpoint_callback = pl.callbacks.ModelCheckpoint(
			filepath=self.output_path,
			monitor=self.hparams.get(
				"checkpoint_monitor", "avg_val_loss"
			),
			mode=self.hparams.get(
				"checkpoint_monitor_mode", "min"
			),
			verbose=self.hparams.get("verbose", True)
		)

		early_stop_callback = pl.callbacks.EarlyStopping(
			monitor=self.hparams.get(
				"early_stop_monitor", "avg_val_loss"
			),
			min_delta=self.hparams.get(
				"early_stop_min_delta", 0.001
			),
			patience=self.hparams.get(
				"early_stop_patience", 3
			),
			verbose=self.hparams.get("verbose", True),
		)

		trainer_params = {
			"checkpoint_callback": checkpoint_callback,
			"early_stop_callback": early_stop_callback,
			"default_save_path": self.output_path,
			"accumulate_grad_batches": self.hparams.get(
				"accumulate_grad_batches", 1
			),
			"gpus": self.hparams.get("n_gpu", 1),
			"max_epochs": self.hparams.get("max_epochs", 100),
			"gradient_clip_val": self.hparams.get(
				"gradient_clip_value", 0
			),
		}
		return trainer_params

	@torch.no_grad()
	def make_submission_frame(self, test_path):
		test_dataset = self._build_dataset(test_path)
		submission_frame = pd.DataFrame(
			index=test_dataset.samples_frame.id,
			columns=["proba", "label"]
		)
		test_dataloader = torch.utils.data.DataLoader(
			test_dataset, 
			shuffle=False, 
			batch_size=self.hparams.get("batch_size", 4), 
			num_workers=self.hparams.get("num_workers", 16))
		for batch in tqdm(test_dataloader, total=len(test_dataloader)):
			preds, _ = self.model.eval().to("cpu")(
				batch["text"], batch["image"]
			)
			submission_frame.loc[batch["id"], "proba"] = preds[:, 1]
			submission_frame.loc[batch["id"], "label"] = preds.argmax(dim=1)
		submission_frame.proba = submission_frame.proba.astype(float)
		submission_frame.label = submission_frame.label.astype(int)
		return submission_frame

