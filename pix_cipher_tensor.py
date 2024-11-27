import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms as T
import matplotlib.pyplot as plt

def block_division(img, block_size=16):
    # (224, 224, 3) -> (16*16, 2*2, 8*8, 3)
    batch_size, channel_num, height, width = img.shape
    block_num_h, block_num_w = height // block_size, width // block_size

    # block division
    block_group = img.reshape(batch_size, channel_num, block_num_h, block_size, block_num_w, block_size)
    block_group = block_group.permute(0, 2, 4, 1, 3, 5)

    block_group = block_group.reshape(batch_size, block_num_h * block_num_w, channel_num, block_size, block_size)

    return block_group


def block_integration(block_group, img_height, img_width):
    # (batch_size, block_num, channel_num, blk_size, blk_size)
    batch_size, block_num, channel_num, block_size, _  = block_group.shape
    block_num_h, block_num_w = img_height // block_size, img_width // block_size

    # block integration
    block_group = block_group.reshape(batch_size, block_num_h, block_num_w, channel_num, block_size, block_size)
    block_group = block_group.permute(0, 3, 1, 4, 2, 5)
    img = block_group.reshape(batch_size, channel_num, img_height, img_width)
    return img

class PixCipher:
    def __init__(self, b_size, pix_shuffle_key, block_shuffle_key):
        self.b_size = b_size
        self.img_size = 224
        self.pix_shuffle_key = pix_shuffle_key
        self.block_shuffle_key = block_shuffle_key

    def encrypt_img(self, img):
        block_group = block_division(img, self.b_size)
        n_batch, n_patch, n_c, patch_size, _ = block_group.shape

        block_group = block_group.reshape(n_batch, n_patch, -1)
        block_group = block_group[:, :, self.pix_shuffle_key] # pixel shuffling
        block_group = block_group[:, self.block_shuffle_key, :] # block shuffling

        block_group = block_group.reshape(n_batch, n_patch, n_c, patch_size, patch_size)
        encrypted_img = block_integration(block_group, self.img_size, self.img_size)

        return encrypted_img


def visualize_img(img_tensor):
    grid = make_grid(img_tensor, nrow=1, padding=2)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()


def main():
    b_size = 16
    b_num = (224 // 16) ** 2
    key_1 = np.random.permutation(b_size ** 2 * 3)
    key_2 = np.random.permutation(b_num)

    key_dict = {'key_1': key_1, 'key_2': key_2}
    np.savez('key_dicts/pix_and_pos.npz', **key_dict)

    img_cipher = PixCipher(16, key_1, key_2)
    transform = T.ToTensor()

    img = Image.open('img.png')
    img = transform(img).unsqueeze(0)
    img = img_cipher.encrypt_img(img)




    visualize_img(img)



if __name__ == '__main__':
    main()
