import torch
import litert_torch
from model import get_model

NUM_CLASSES = 6

# Load trained model
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Dummy input
dummy = torch.randn(1, 3, 224, 224)

# Convert to LiteRT
edge_model = litert_torch.convert(model, (dummy,))

# Export TFLite
edge_model.export("model.tflite")

print("TFLite model created successfully.")
