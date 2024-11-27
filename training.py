import json
import os
import time

import timm
import torch
import tqdm
import numpy as np

from timm.scheduler import CosineLRScheduler
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from ProgressMonitors.training_monitor import TrainingMonitor
from datasets.CIFAR10 import CIFAR10_handler
from pix_cipher_tensor import PixCipher
from models.Crypto_ViT import CryptoViT_timm, CryptoLoRaViT_timm
from models.LoRa_ViT import LoRA_ViT_timm
from models.Vanilla_ViT import VanillaViT_timm
from trainers.base_trainer_v2 import Trainer


def check_trainable_layers(model):
    trainable_layers = []

    for name, module in model.named_modules():
        # 检查模块中的任意一个参数是否 requires_grad
        has_trainable_params = any(
            param.requires_grad for param in module.parameters(recurse=False)
        )
        if has_trainable_params:
            trainable_layers.append((name, module))

    return trainable_layers

def count_trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {num_params / 2 ** 20:.3f}M")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {num_params / 2 ** 20:.3f}M")


def save_checkpoint(ckpt_state, ckpt_dir, **kwargs):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, 'final_ckpt.pth')
    torch.save(ckpt_state, save_path)


def training_loop(trainer:Trainer, data_handler, total_epochs, lora_used):
    train_loader = data_handler.train_loader()
    val_loader = data_handler.val_loader()
    test_loader = data_handler.test_loader()

    for epoch in range(total_epochs):
        train_progress = tqdm.tqdm(train_loader)
        for i, batch_data in enumerate(train_progress):
            trainer.training_step_GA(batch_data, i, len(train_loader))
            train_progress.set_description(f"Training: Epoch {epoch + 1} / {total_epochs}")
        trainer.training_epoch_callback(epoch)

        # validation step
        for i, batch_data in enumerate(val_loader):
            trainer.validation_step(batch_data)
        val_acc_updated = trainer.validation_epoch_callback()

        if val_acc_updated:
            for i, batch_data in enumerate(test_loader):
                trainer.testing_step(batch_data)
            trainer.testing_epoch_callback()
            trainer.save_checkpoint('best_val_epoch')
            if lora_used:
                trainer.model.save_lora_parameters('ckpts/best_val_lora_weights.safetensors')

        trainer.monitor.save_to_json('last_epoch')
        trainer.save_checkpoint('last_epoch')
        if lora_used:
            trainer.model.save_lora_parameters('ckpts/full_training_lora_weights.safetensors')


def main(config):
    torch.cuda.empty_cache()
    start_time = time.time()

    model_name = config['model_name']
    fine_tuning = config['ft_mode']
    n_classes = config['n_classes']
    encrypted_training = config['encrypted_training']

    dataset_name = config['dataset_name']
    val_set_size = config['val_set_size']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    mix_up = config['mix_up']
    accumulation_steps = config['accumulation_steps']

    lr = config['learning_rate']
    t_warmup = config['t_warmup']
    weight_decay = config['weight_decay']
    random_seed = config['random_seed']
    n_epochs = config['n_epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # linear probing
    if fine_tuning == 'lp':
        print('Use Vanilla ViT for linear probing fine-tuning...')
        model = VanillaViT_timm(model_name, n_classes, frozen_backbone=True)
        lora_used = False

    # end-to-end fine-tuning
    elif fine_tuning == 'e2e':
        print('Use Vanilla ViT for end-to-end fine-tuning...')
        model = VanillaViT_timm(model_name, n_classes, frozen_backbone=False)
        lora_used = False

    # fine-tuning with lora
    elif fine_tuning == 'lora':
        print('Use LoRa ViT..')
        model = timm.create_model(model_name, pretrained=True)
        model = LoRA_ViT_timm(vit_model=model, r=4, alpha=8, num_classes=n_classes)
        lora_used = True

    else:
        print("Wrong model name!")
        exit()

    if encrypted_training:
        key_dict = np.load('key_dicts/pix_and_pos.npz')
        img_cipher = PixCipher(16, key_dict['key_1'], key_dict['key_2'])

        if fine_tuning == 'lora':
            model = CryptoLoRaViT_timm(model, key_dict, img_cipher)
        else:
            model = CryptoViT_timm(model, key_dict, img_cipher)

        model.load_encrypted_weights()

    count_trainable_parameters(model)

    if dataset_name == 'CIFAR10':
        print("Use CIFAR10_plain dataset..")
        dataset_dir = 'datasets/downloaded_datasets/CIFAR10_zip'
        data_handler = CIFAR10_handler(dataset_dir, val_set_size, batch_size, num_workers)

    else:
        print("Wrong dataset name!")
        exit()


    if encrypted_training:
        record_dir = f'./ckpts/{fine_tuning}_encrypted'
    else:
        record_dir = f'./ckpts/{fine_tuning}_plain'
    os.makedirs(record_dir, exist_ok=False)

    torch.manual_seed(random_seed)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    lr_scheduler = CosineLRScheduler(optimizer=optimizer, t_initial=n_epochs,
                                     warmup_t=t_warmup, warmup_lr_init=0.01 * lr)
    training_monitor = TrainingMonitor(data_handler)
    trainer = Trainer(model, data_handler, optimizer, lr_scheduler, criterion, training_monitor,
                      device, n_epochs, mix_up, accumulation_steps)

    training_loop(trainer, data_handler, n_epochs, lora_used)
    print(f"training consumed time: {(time.time() - start_time):.2f} s")

    # copy the config for recording
    with open(f'{record_dir}/training_config.json', "w") as f:
        json.dump(config, f, indent=4)


if __name__ == '__main__':
    with open(f'script/training/config_CIFAR10_encrypted.json') as config_file:
        training_config = json.load(config_file)

    main(training_config)


