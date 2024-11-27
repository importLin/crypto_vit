from torch.utils.data import Dataset, random_split, DataLoader


def split_train_val(train_set:Dataset, val_set_size:float):
    imgs_total_num = train_set.__len__()
    val_imgs_num = int(imgs_total_num * val_set_size)
    return random_split(train_set, [imgs_total_num - val_imgs_num, val_imgs_num])


class Data_handler:
    def __init__(self, train_set:Dataset, test_set:Dataset, val_set, batch_size, num_workers):
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set

        self.batch_size = batch_size
        self.val_batch_size = batch_size * 4
        self.num_workers = num_workers

    def train_loader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )


    def val_loader(self):
        if self.val_set is None:
            return None
        else:
            return DataLoader(
                self.val_set,
                self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False
            )


    def test_loader(self):
        return DataLoader(
            self.test_set,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )

    def get_samples_num(self):
        return len(self.train_set), len(self.val_set), len(self.test_set)