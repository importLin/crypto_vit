import os

import torch
from timm.data.mixup import Mixup

from ProgressMonitors.training_monitor import TrainingMonitor
from utils.early_stopping import EarlyStopping


class Trainer:
    def __init__(self, model, data_handler, optimizer, lr_scheduler, criterion, monitor:TrainingMonitor, device, n_epochs,
                 mix_up, accumulation_steps):
        super(Trainer,self).__init__()

        self.model = model.to(device)
        self.data_handler = data_handler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.monitor = monitor

        self.device = device
        self.n_epochs = n_epochs
        self.ckpt_state = {}

        self.early_stopping = EarlyStopping(patience=10, verbose=True)
        if mix_up == 0:
            print("no mix up applied!")
            self.mix_up = None

        else:
            self.mix_up = Mixup(mixup_alpha=mix_up, num_classes=3)

        self.accumulation_steps = accumulation_steps


    def training_step(self, batch_data):
        self.model.train()
        # Forward pass
        img_batch, label_batch= batch_data
        img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)

        if self.mix_up:
            img_batch, label_batch = self.mix_up(img_batch, label_batch)

        if self.mix_up:
            img_batch, label_batch = self.mix_up(img_batch, label_batch)

        # self.optimizer.zero_grad()
        logits = self.model(img_batch)
        loss = self.criterion(logits, label_batch)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        info_dict = {'stage':'train', 'n_samples': img_batch.size(0), 'n_correct': 0, 'batch_loss': loss.item()}
        self.monitor.update_step_info(info_dict)

    # Gradient Accumulation
    def training_step_GA(self, batch_data, current_step, last_step):
        self.model.train()
        # Forward pass
        img_batch, label_batch= batch_data
        img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)
        if self.mix_up:
            img_batch, label_batch = self.mix_up(img_batch, label_batch)

        logits = self.model(img_batch)
        loss = self.criterion(logits, label_batch)
        if (current_step + 1) != last_step:
            loss = loss / self.accumulation_steps
            loss_for_record = loss.item() * self.accumulation_steps
        else:
            loss_for_record = loss.item()
        loss.backward()

        if ((current_step + 1) % self.accumulation_steps == 0) or (current_step + 1 == last_step):
            self.optimizer.step()
            self.optimizer.zero_grad()

        info_dict = {'stage':'train', 'n_samples': img_batch.size(0), 'n_correct': 0, 'batch_loss': loss_for_record}
        self.monitor.update_step_info(info_dict)


    def training_epoch_callback(self, epoch):
        self.lr_scheduler.step(epoch)
        self.monitor.update_epoch_info('train')

    def validation_step(self, batch_data):
        self.model.eval()
        with torch.no_grad():
            img_batch, label_batch = batch_data
            img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)

            logits = self.model(img_batch)
            loss = self.criterion(logits, label_batch)
            preds = torch.argmax(logits, dim=1)

            batch_correct = (preds == label_batch).sum().item()

            info_dict = {'stage':'val', 'n_samples': img_batch.size(0), 'n_correct': batch_correct, 'batch_loss': loss.item()}
            self.monitor.update_step_info(info_dict)


    def validation_epoch_callback(self):
        val_acc_updated = False
        self.monitor.update_epoch_info('val')
        if self.monitor.is_val_best():
            self.monitor.update_val_best()
            self.ckpt_state.update({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'lr_scheduler_state': self.lr_scheduler.state_dict()
            })
            self.save_checkpoint('best_val_epoch')
            print(f"The best val accuracy has been updated to {self.monitor.val_best_acc:.4f}")
            val_acc_updated = True

        # self.early_stopping(self.monitor.val_epoch_losses[-1])

        return val_acc_updated


    def testing_step(self, batch_data):
        self.model.eval()
        with torch.no_grad():
            img_batch, label_batch = batch_data
            img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)

            logits = self.model(img_batch)
            preds = torch.argmax(logits, dim=1)
            loss = self.criterion(logits, label_batch)

            batch_correct = (preds == label_batch).sum().item()

            info_dict = {'stage':'test', 'n_samples': img_batch.size(0), 'n_correct': batch_correct, 'batch_loss': loss.item()}
            self.monitor.update_step_info(info_dict)


    def testing_epoch_callback(self):
        self.monitor.update_epoch_info('test')


    def restart_training(self):
        print("Loading state from trained ckpts...")
        self.start_epoch = self.ckpt_state['last_epoch']
        self.val_best_acc = self.ckpt_state['val_best_acc']
        self.model.load_state_dict(self.ckpt_state['model_state'])
        self.optimizer.load_state_dict(self.ckpt_state['optimizer_state'])
        self.lr_scheduler.load_state_dict(self.ckpt_state['lr_scheduler_state'])


    def save_checkpoint(self, mode):
        if mode == 'best_val_epoch':
            torch.save(self.ckpt_state, os.path.join('./ckpts', f'best_val_state.pth'))

        elif mode == 'last_epoch':
            torch.save(self.ckpt_state, os.path.join('./ckpts', f'full_training_state.pth'))
        else:
            return





