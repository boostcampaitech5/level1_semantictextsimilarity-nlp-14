import argparse

import pandas as pd

from tqdm.auto import tqdm
import numpy as np
import random
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
import wandb
import nlpaug.augmenter.word as naw


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=128)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = [item[text_column] for text_column in self.text_columns]
            aug = naw.SynonymAug()
            augmented_text = aug.augment(text)
            text_augmented = ['[SEP]'.join([augmented_text[i], text[i]]) for i in range(len(text))]
            if not text_augmented:
                continue
        
            outputs = self.tokenizer(text_augmented, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data



    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, weight_decay, warmup_steps, total_steps, loss_func):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 손실함수를 호출합니다.
        if loss_func == "MSE":
            self.loss_func = torch.nn.MSELoss()
        elif loss_func == "L1":
            self.loss_func = torch.nn.L1Loss()
        elif loss_func == "Huber":
            self.loss_func = torch.nn.HuberLoss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # lr scheduler를 이용해 warm-up stage 추가
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps = self.total_steps
        )
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'val_loss',
                }
            ]
        )




if __name__ == '__main__':
    # seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=6, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='~/data/train.csv')
    parser.add_argument('--dev_path', default='~/data/dev.csv')
    parser.add_argument('--test_path', default='~/data/dev.csv')
    parser.add_argument('--predict_path', default='~/data/test.csv')
    parser.add_argument('--weight_decay', default=0.01)
    parser.add_argument('--warm_up_ratio', default=0.3)
    parser.add_argument('--loss_func', default="MSE")
    parser.add_argument('--run_name', default="roberta_large_001")
    args = parser.parse_args()

    sweep_config = {
        'method': 'random', # random: 임의의 값의 parameter 세트를 선택
        'parameters': {
            'learning_rate':{
                'values':[1e-5, 2e-5, 3e-5, 5e-5]
            },
            'max_epoch':{
                'values':[3, 4, 5]
            },
            'batch_size':{
                'values':[8, 16]
            },
            'weight_decay':{
                'values':[0., 0.01, 0.1]
            },
            'warm_up_ratio':{
                'values':[0., 0.1, 0.2, 0.6]
            },
            'loss_func':{
                'values':["MSE", "Huber"]
            }
        },
        'metric': {
            'name':'val_pearson',
            'goal':'maximize'
        }
    }


    # Sweep을 통해 실행될 학습 코드를 작성합니다.
    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config

        dataloader = Dataloader(args.model_name, config.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        warmup_steps = int((9324 // config.batch_size + (9324 % config.batch_size != 0)) * config.warm_up_ratio)
        model = Model(
            args.model_name,
            config.learning_rate,
            config.weight_decay,
            warmup_steps,
            config.loss_func
        )
        wandb_logger = WandbLogger(project="sts-04-12-001")

        trainer = pl.Trainer(precision="16-mixed", accelerator='gpu', max_epochs=config.max_epoch, logger=wandb_logger, log_every_n_steps=1, gradient_clip_val=1.)
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
    
    # Sweep 생성

    # sweep_id = wandb.sweep(
    #     sweep=sweep_config,     # config 딕셔너리를 추가합니다.
    #     project='sts-04-12-001'  # project의 이름을 추가합니다.
    # )
    # wandb.agent(
    #     sweep_id=sweep_id,      # sweep의 정보를 입력하고
    #     function=sweep_train,   # train이라는 모델을 학습하는 코드를
    #     count=40                 # 총 3회 실행해봅니다.
    # )
    
    # wandb logger
    wandb.init()
    wandb.run.name = args.run_name
    wandb_logger = WandbLogger(project="sts-004")

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    total_steps = (9324 // args.batch_size + (9324 % args.batch_size != 0)) * args.max_epoch
    warmup_steps = int((9324 // args.batch_size + (9324 % args.batch_size != 0)) * args.warm_up_ratio)
    model = Model(
        args.model_name,
        args.learning_rate,
        args.weight_decay,
        warmup_steps,
        total_steps,
        args.loss_func
    )
    # print(model)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(precision="16-mixed", accelerator='gpu', max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=1)

    # use Tuner to get optimized batch size
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model=model, datamodule=dataloader, mode="binsearch")

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()
    
    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')