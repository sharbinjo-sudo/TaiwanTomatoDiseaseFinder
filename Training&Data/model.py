import timm
import torch.nn as nn

def get_model(num_classes):

    # Create pretrained ViT
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True
    )

    # Replace classifier head properly
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    # 1️⃣ Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False

    # 2️⃣ Unfreeze classifier head
    for param in model.head.parameters():
        param.requires_grad = True

    # 3️⃣ Unfreeze LAST 4 transformer blocks (controlled fine-tuning)
    for block in model.blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True

    return model
