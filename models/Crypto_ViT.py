import torch
from torch import nn


class CryptoViT_timm(nn.Module):
    def __init__(self, timm_vit, key_dict, img_cipher):
        super(CryptoViT_timm, self).__init__()
        self.model = timm_vit
        self.key_dict = key_dict
        self.img_cipher = img_cipher

        self.patch_emb_weight = self.extract_weight('model.patch_embed.proj.weight')
        self.pos_emb_weight = self.extract_weight('model.pos_embed')
        self.is_encrypted = False

    def forward(self, x):
        if self.is_encrypted:
            x = self.img_cipher.encrypt_img(x)
        return self.model(x)

    def extract_weight(self, weight_name):
        weight = None
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name == weight_name:
                    weight = param.clone()
                    break
        if weight is not None:
            return {'name': weight_name, 'weight': weight}
        else:
            ValueError('No weight found for {}'.format(weight_name))

    def encrypt_patch_emb(self, secret_order):
        d, c, p_size, _ = self.patch_emb_weight['weight'].shape

        encrypted_weight = self.patch_emb_weight['weight'].reshape(d, -1)
        encrypted_weight = encrypted_weight[..., secret_order]
        self.patch_emb_weight['weight'] = encrypted_weight.reshape(d, c, p_size, p_size)

    def encrypt_pos_emb(self, secret_order):
        # the shuffling does not change the pos embd of cls token
        self.pos_emb_weight['weight'][:, 1:, :] = self.pos_emb_weight['weight'][:, 1:, :][:, secret_order, :]

    def load_encrypted_weights(self):
        self.encrypt_patch_emb(self.key_dict['key_1'])
        self.encrypt_pos_emb(self.key_dict['key_2'])

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name == self.patch_emb_weight['name']:
                    param.copy_(self.patch_emb_weight['weight'])
                    param.requires_grad = False

                if name == self.pos_emb_weight['name']:
                    param.copy_(self.pos_emb_weight['weight'])
        self.is_encrypted = True


class CryptoLoRaViT_timm(CryptoViT_timm):
    def __init__(self, timm_lora_vit, key_dict, img_cipher):
        super(CryptoLoRaViT_timm, self).__init__(timm_lora_vit, key_dict, img_cipher)
        self.patch_emb_weight = self.extract_weight('lora_vit.patch_embed.proj.weight')
        self.pos_emb_weight = self.extract_weight('lora_vit.pos_embed')

    def save_lora_parameters(self, save_path):
        self.model.save_lora_parameters(save_path)



def main():
    pass
    # key_dict = np.load('pix_only.npz')
    # model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # crypto_vit = CryptoViT_timm(model, key_dict=key_dict)
    # crypto_vit.load_encrypted_weights()
    #
    # # timm_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    # # timm_lora_vit = LoRA_ViT_timm(vit_model=timm_vit, r=4, alpha=8, num_classes=10)
    # # crypto_lora_vit = CryptoLoRaViT_timm(timm_lora_vit, key_dict=key_dict)
    # # crypto_lora_vit.load_encrypted_weights()



if __name__ == '__main__':
    main()
