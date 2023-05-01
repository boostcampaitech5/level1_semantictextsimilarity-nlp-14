import argparse
import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from itertools import chain
from seed import *  # seed setting module

## config
import config

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내옴
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용
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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column]
                                for text_column in self.text_columns])
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding='max_length', truncation=True)
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
            self.train_inputs, self.train_targets = self.preprocessing(
                train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # 검증데이터 세팅
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
        """_summary_

        토크나이징 된 데이터를 샘플링

        Returns:
            _type_: _description_
        """
        origin_data = pd.DataFrame({'data': self.train_inputs, 'label': list(
            chain.from_iterable(self.train_targets))})
        # 소수점 첫째자리 짝수면 600개, 홀수면 60개 샘플링
        train_data = pd.concat([origin_data[origin_data.label == i/10].sample(600, replace=True) for i in range(0, 51, 2)] +
                               [origin_data[origin_data.label == i/10].sample(60, replace=True) for i in range(5, 46, 10)])
        self.train_dataset = Dataset(train_data.data.tolist(),
                                     [[i] for i in train_data.label])
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name,
        lr,
        weight_decay,
        loss_func,
        warmup_steps,
        total_steps
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # 사용할 모델을 호출
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 손실함수를 호출
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

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # warmup stage 있는 경우
        if self.warmup_steps is not None:
            scheduler = transformers.get_inverse_sqrt_schedule(
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps
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
        # warmup stage 없는 경우
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.96)
            return [optimizer], [scheduler]


# 모델 저장을 위한 class
class CustomModelCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            current = monitor_candidates.get(self.monitor)
            # added
            if torch.isnan(current) or current < 0.93:
                return
            ###
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)


if __name__ == '__main__':
    # set seed
    seed = get_seed()
    set_seed(*seed)
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--model_name', default="snunlp/KR-ELECTRA-discriminator", type=str)
    # parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--max_epoch', default=10, type=int)
    # parser.add_argument('--shuffle', default=True)
    # parser.add_argument('--learning_rate', default=2e-5, type=float)
    # parser.add_argument(
    #     '--train_path', default='~/data/train_sentence_swap.csv')
    # parser.add_argument('--dev_path', default='~/data/dev.csv')
    # parser.add_argument('--test_path', default='~/data/dev.csv')
    # parser.add_argument('--predict_path', default='~/data/test.csv')
    # parser.add_argument('--weight_decay', default=0.01)
    # parser.add_argument('--warm_up_ratio', default=0.3)
    # parser.add_argument('--loss_func', default="MSE")
    # parser.add_argument('--run_name', default="001")
    # parser.add_argument('--project_name', default="STS_snunlp_9250")
    # parser.add_argument('--entity', default=None)   # wandb team name
    # args = parser.parse_args()

    # refactoring
    args = config.config
    print(args.project_name)
    print(f"{args.loss_func}_{args.learning_rate}_{args.batch_size}_{args.weight_decay}_{args.max_epoch}_steplr_seed:{'_'.join(map(str,seed))}")

    # actual model train
    # wandb logger
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=f"{args.loss_func}_{args.learning_rate}_{args.batch_size}_{args.weight_decay}_{args.max_epoch}_steplr_seed:{'_'.join(map(str,seed))}",
        entity=args.entity
    )

    # # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

    total_steps = warmup_steps = None
    if args.warm_up_ratio is not None:
        total_steps = (15900 // args.batch_size + (15900 %
                       args.batch_size != 0)) * args.max_epoch
        warmup_steps = int((15900 // args.batch_size + (15900 %
                           args.batch_size != 0)) * args.warm_up_ratio)

    model = Model(
        args.model_name,
        args.learning_rate,
        args.weight_decay,
        args.loss_func,
        warmup_steps,
        total_steps,
    )

    # model = torch.load('model.pt')

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        precision="16-mixed",                   # 16-bit mixed precision
        accelerator='cpu',                      # GPU 사용
        # dataloader를 매 epoch마다 reload해서 resampling
        reload_dataloaders_every_n_epochs=1,
        max_epochs=args.max_epoch,            # 최대 epoch 수
        logger=wandb_logger,                    # wandb logger 사용
        log_every_n_steps=1,                    # 1 step마다 로그 기록
        val_check_interval=0.25,                # 0.25 epoch마다 validation
        check_val_every_n_epoch=1,              # val_check_interval의 기준이 되는 epoch 수
        callbacks=[
            # learning rate를 매 step마다 기록
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(                      # validation pearson이 8번 이상 개선되지 않으면 학습을 종료
                'val_pearson',
                patience=8,
                mode='max',
                check_finite=False
            ),
            CustomModelCheckpoint(
                './save/',
                'snunlp_MSE_002_{val_pearson:.4f}',
                monitor='val_pearson',
                save_top_k=1,
                mode='max'
            )
        ]
    )

    # use Tuner to get optimized batch size
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model=model, datamodule=dataloader, mode="binsearch")

    # Train part
    # trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # # 학습이 완료된 모델을 저장합니다.
    # torch.save(model, 'model.pt')
