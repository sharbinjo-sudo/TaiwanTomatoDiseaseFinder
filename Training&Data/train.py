import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_loaders
from model import get_model

ROOT = "/workspace/taiwan/Preprocessed data"
EPOCHS = 50
LR = 5e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dl, val_dl, test_dl, num_classes = get_loaders(ROOT)

model = get_model(num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

best_acc = 0.0
patience = 8
counter = 0

for epoch in range(EPOCHS):

    model.train()
    correct, total = 0, 0

    for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    v_correct, v_total = 0, 0

    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            v_correct += (pred == y).sum().item()
            v_total += y.size(0)

    val_acc = 100 * v_correct / v_total

    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model.pth")
        counter = 0
        print("Saved best model")
    else:
        counter += 1

    print(f"Epoch {epoch+1}: Train={train_acc:.2f} | Val={val_acc:.2f}")

    if counter >= patience:
        print("Early stopping triggered")
        break

print("Training complete.")
