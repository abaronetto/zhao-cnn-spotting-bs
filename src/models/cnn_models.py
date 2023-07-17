import torch
import torch.nn as nn
import os
import sys
sys.path.append("../")
import pytorch_lightning as pl
import torch.distributed as dist
from torchmetrics.classification import Precision, Recall, F1Score, Accuracy

class FocalLoss(pl.LightningModule):

    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        super().__init__()
        self.weight = weight.to(self.device) if weight is not None else None  # TODO check weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = torch.functional.F.log_softmax(input_tensor.to(self.device), dim=-1)
        prob = torch.exp(log_prob)
        w = self.weight.float().to(self.device) if self.weight is not None else None
        return torch.functional.F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            torch.argmax(target_tensor.long(), dim=1).to(self.device),
            weight=w,
            reduction=self.reduction
        )


class CNN(pl.LightningModule):
    def __init__(self, label_dim=527, scheduler={}, loss='CE', weight=None, exp_dir='', warmup=False):
        super(CNN, self).__init__()
        # self.save_hyperparameters()

        # scheduler options
        self.lr,  self.lrscheduler_decay = scheduler['lr'], scheduler['lrscheduler_decay']

        self.warmup = warmup

        # loss_fn
        if loss == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss == 'FL':
            if weight is not None:
                weight = weight
            self.loss_fn = FocalLoss(weight=weight, gamma=2).to(self.device)

        # metrics
        self.prec = Precision(dist_sync_on_step=False, multiclass=False)
        self.rec = Recall(dist_sync_on_step=False, multiclass=False)
        self.f1 = F1Score(dist_sync_on_step=False, multiclass=False)
        # self.acc = Accuracy(dist_sync_on_step=False, multiclass=True)

        self.path_results = exp_dir

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')

        self.fc = nn.Linear(768, label_dim)

    def forward(self, x, nframes=1056):

        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

    def configure_optimizers(self):
        # Set up the optimizer
        trainables = [p for p in self.parameters() if p.requires_grad]
        print(
            'Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in self.parameters()) / 1e6))
        print(
            'Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

        optimizer = torch.optim.Adadelta(
            [{'params': trainables, 'lr': self.lr}],
            weight_decay=1e-4, rho=self.lrscheduler_decay)

        return [optimizer]

    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,  on_tpu=False,
            using_native_amp=False, using_lbfgs=False):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step <= 1000 and self.trainer.global_step % 50 == 0 and self.warmup:
            warm_lr = (self.trainer.global_step / 1000) * self.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_lr
            print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

    def training_step(self, batch, batch_idx):
        x, y, data = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, data = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        data[:, -3] = y_hat[:, 0]
        data[:, -2] = y_hat[:, 1]
        data[:, -1] = torch.argmax(y_hat, dim=1)
        self.prec(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.rec(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.f1(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.log("val_loss", val_loss, on_step=False,prog_bar=True, on_epoch=True)
        return [y_hat, y, data]

    def test_epoch_end(self, outputs) -> None:
        data_pred = torch.cat([d[2] for d in outputs], dim=0)
        gather_data = [torch.zeros_like(data_pred) for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather(gather_data, data_pred)
        if dist.get_rank() == 0:
            gather_data = torch.cat(gather_data, dim=0)
            assert all(gather_data[:, -1] != -1)
            # save data for metrics
            torch.save(gather_data, self.path_results + f'/predictions_epoch{self.current_epoch}.pt')
        self.log('precision', self.prec, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.log('recall', self.rec, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.log('f1', self.f1, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        dist.barrier()

    def predict_step(self, batch, batch_idx):
        x, y, data = batch
        pred = self(x)
        return pred

    def validation_step(self, batch, batch_idx):
        x, y, data = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        data[:, -3] = y_hat[:, 0]
        data[:, -2] = y_hat[:, 1]
        data[:, -1] = torch.argmax(y_hat, dim=1)
        self.prec(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.rec(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.f1(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.log("val_loss", val_loss, on_step=False,prog_bar=True, on_epoch=True)
        return [y_hat, y, data]

    def validation_epoch_end(self, outputs) -> None:
        data_pred = torch.cat([d[2] for d in outputs], dim=0)
        gather_data = [torch.zeros_like(data_pred) for _ in range(dist.get_world_size())]
        dist.barrier()
        dist.all_gather(gather_data, data_pred)
        if dist.get_rank() == 0:
            gather_data = torch.cat(gather_data, dim=0)
            assert all(gather_data[:, -1] != -1)
            # save data for metrics
            torch.save(gather_data, self.path_results + f'/predictions_epoch{self.current_epoch}.pt')
        self.log('precision', self.prec, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.log('recall', self.rec, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.log('f1', self.f1, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        dist.barrier()
