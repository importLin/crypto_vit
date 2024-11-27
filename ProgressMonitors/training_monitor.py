import json
import os

class TrainingMonitor:
    def __init__(self, data_handler):
        self.train_samples_num = len(data_handler.train_set)
        self.val_samples_num = len(data_handler.val_set)
        self.test_samples_num = len(data_handler.test_set)

        self.current_epoch = 0
        self.val_best_epoch = 0
        self.test_best_epoch = 0

        # status monitor
        self.train_step_losses = []
        self.train_epoch_losses = []
        self.train_correct_num = 0.0
        self.train_current_acc = 0.0

        self.val_step_losses = []
        self.val_epoch_losses = []
        self.val_correct_num = 0.0
        self.val_current_acc = 0.0
        self.val_best_acc = 0

        self.test_correct_num = 0.0
        self.test_current_acc = 0.0
        self.test_best_acc = 0

    def update_step_info(self, info_dict):
        stage = info_dict['stage']
        n_samples = info_dict['n_samples']
        n_correct = info_dict['n_correct']
        batch_loss = info_dict['batch_loss']

        if stage == 'train':
            self.train_step_losses.append(n_samples * batch_loss)
            self.train_correct_num += n_correct

        elif stage == 'val':
            self.val_step_losses.append(n_samples * batch_loss)
            self.val_correct_num += n_correct


        elif stage == 'test':
            self.test_correct_num += n_correct

    def update_epoch_info(self, stage):
        if stage == 'train':
            self.train_epoch_losses.append(sum(self.train_step_losses) / self.train_samples_num)
            self.train_step_losses = []
            self.train_current_acc = self.train_correct_num / self.train_samples_num
            self.train_correct_num = 0.0
            self.current_epoch += 1

            print(f'Epoch: {self.current_epoch:<3}\t'
                  f'Train loss: {self.train_epoch_losses[-1]:<8.3f}')


        elif stage == 'val':
            self.val_epoch_losses.append(sum(self.val_step_losses) / self.val_samples_num)
            self.val_step_losses = []
            self.val_current_acc = self.val_correct_num / self.val_samples_num
            self.val_correct_num = 0.0

            print(f'Epoch: {self.current_epoch:<3}\t'
                  f'Val loss: {self.val_epoch_losses[-1]:<8.4f}'
                  f'Val acc: {self.val_current_acc:<8.4f}'
                  f'Val best acc (test acc): {self.val_best_acc:.4f} ({self.test_current_acc:.4f})')

        elif stage == 'test':
            self.test_current_acc = self.test_correct_num / self.test_samples_num
            self.test_correct_num = 0.0

            print(f'Epoch: {self.current_epoch:<3}\t'
                  f'Last test acc: {self.test_current_acc:<8.4f}')


    def is_val_best(self):
        return self.val_current_acc > self.val_best_acc

    def update_val_best(self):
        if self.val_current_acc > self.val_best_acc:
            self.val_best_acc = self.val_current_acc
            self.save_to_json('best_val_epoch')


    def save_to_json(self, mode, log_dir='ckpts'):
        if mode == 'best_val_epoch':
            log_filename = os.path.join(log_dir, 'best_val_log.json')
        elif mode == 'last_epoch':
            log_filename = os.path.join(log_dir, 'full_training_log.json')
        else:
            return

        record = {
            'current_epoch': self.current_epoch,
            'val_current_acc':self.val_current_acc,
            'test_current_acc': self.test_current_acc,
            'train_epoch_losses': self.train_epoch_losses,
            'val_epoch_losses': self.val_epoch_losses
        }

        # save the dict as a json file
        with open(log_filename, "w") as f:
            json.dump(record, f, indent=4)


