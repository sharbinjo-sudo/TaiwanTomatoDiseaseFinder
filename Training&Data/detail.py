from dataset import get_loaders

ROOT = r"C:\Users\sharb\Documents\TaiwanTomatoDiseaseFinder\Training&Data"

train_dl, val_dl, test_dl, num_classes = get_loaders(ROOT)

print("Classes:", train_dl.dataset.classes)
