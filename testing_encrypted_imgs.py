import os.path
import os.path

import numpy as np
import timm
import torch
from tqdm import tqdm

from datasets.CIFAR10 import CIFAR10_handler
from pix_cipher_tensor import PixCipher
from models.Crypto_ViT import CryptoViT_timm, CryptoLoRaViT_timm
from models.LoRa_ViT import LoRA_ViT_timm
from models.Vanilla_ViT import VanillaViT_timm


def main():
    ft_type = 'lora'
    n_classes = 10
    val_set_size = 0
    dataset_dir = './datasets/downloaded_datasets/CIFAR10_zip' # 94.53
    batch_size = 1024
    num_workers = 8

    ckpt_dir = f'ckpts/{ft_type}_encrypted'
    state_dict = torch.load(os.path.join(ckpt_dir, 'best_val_state.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(3407)

    key_dict = np.load('key_dicts/pix_and_pos.npz')
    img_cipher = PixCipher(16, key_dict['key_1'], key_dict['key_2'])

    if ft_type in ['lp', 'e2e']:
        print('Using lr or e2e')
        model = VanillaViT_timm('vit_base_patch16_224', n_classes)
        model_state_dict = {key.replace("model.", "", 1): value for key, value in state_dict['model_state'].items()}
        model.load_state_dict(model_state_dict)
        model = CryptoViT_timm(model, key_dict, img_cipher)

    elif ft_type == 'lora':
        print('Using lora')
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = LoRA_ViT_timm(vit_model=model, r=4, alpha=8, num_classes=n_classes)

        model_state_dict = {key.replace("model.", "", 1): value for key, value in state_dict['model_state'].items()}
        model.load_state_dict(model_state_dict)
        model.load_lora_parameters(os.path.join(ckpt_dir, 'best_val_lora_weights.safetensors'))
        model = CryptoLoRaViT_timm(model, key_dict, img_cipher)

    else:
        print("Wrong model name!")
        exit()

    model.is_encrypted = True
    model = model.to(device)

    data_handler = CIFAR10_handler(dataset_dir, val_set_size, batch_size, num_workers)
    test_loader = data_handler.test_loader()

    test_correct_accumulation = 0
    model.eval()
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(test_loader)):
            img_batch, label_batch = batch_data
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            logits = model(img_batch)
            preds = torch.argmax(logits, dim=1)
            batch_correct = (preds == label_batch).sum().item()
            test_correct_accumulation += batch_correct

    print(f'accuracy : {test_correct_accumulation / len(data_handler.test_set)}')



if __name__ == '__main__':
    main()

