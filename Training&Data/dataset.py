import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

IMG_SIZE = 224
BATCH_SIZE = 32

train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(0.3, 0.3, 0.3, 0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

def get_loaders(root):

    train_ds = ImageFolder(root + "/train", transform=train_tfms)
    val_ds   = ImageFolder(root + "/val", transform=val_tfms)
    test_ds  = ImageFolder(root + "/test", transform=val_tfms)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_dl, val_dl, test_dl, len(train_ds.classes)
