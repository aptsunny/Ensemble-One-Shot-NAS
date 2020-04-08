import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as data

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def get_dataset(cls, cutout_length=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.Resize(256), # cifar->imagenet
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip()
    ]

    normalize = [
        transforms.Resize(224), # cifar->imagenet
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose(transf + normalize + cutout)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid

def get_train_dataprovider(batch_size, *, num_workers, use_gpu):
    dataset_train, dataset_valid = get_dataset("cifar10")
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=batch_size,
                                               num_workers=num_workers)
    return DataIterator(train_loader)

def get_val_dataprovider(batch_size, *, num_workers, use_gpu):
    dataset_train, dataset_valid = get_dataset("cifar10")
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=batch_size,
                                               num_workers=num_workers)
    return DataIterator(valid_loader)


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


    print(len(train_dataset))
    print(train_dataset[np.random.randint(len(train_dataset))])


    print(len(valid_dataset))
    print(valid_dataset[np.random.randint(len(valid_dataset))])

    use_gpu = False
    train_batch_size = 128
    valid_batch_size = 200

    train_dataprovider = get_train_dataprovider(train_batch_size, use_gpu=use_gpu, num_workers=3)
    val_dataprovider = get_val_dataprovider(valid_batch_size, use_gpu=use_gpu, num_workers=2)

    train_data = train_dataprovider.next()
    val_data = val_dataprovider.next()

    print(train_data[0].mean().item())
    print(val_data[0].mean().item())

    from IPython import embed
    embed()

if __name__ == '__main__':
    main()
