import torch
from voxws import model, util

# Specify the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the model
fws_model = model.load(encoder_name="base", language="en", device=device)

# Prepare support set
support_examples = ["./test_clips/aandachtig.wav", "./test_clips/stroom.wav",
    "./test_clips/persbericht.wav", "./test_clips/klinkers.wav",
    "./test_clips/zinsbouw.wav"]

classes = ["aandachtig", "stroom", "persbericht", "klinkers", "zinsbouw"]
int_indices = list(range(len(classes)))

support = {
    "paths": support_examples,
    "classes": classes,
    "labels": torch.tensor(int_indices),
}
support["audio"] = torch.stack([util.load_clip(path) for path in support["paths"]])
support = util.batch_device(support, device=device)

# Prepare query set
query = {
    "paths": ["./test_clips/query_klinkers.wav",
              "./test_clips/query_stroom.wav",
              "test_clips/query_klinkers.wav",
              
              ]
}
query["audio"] = torch.stack([util.load_clip(path) for path in query["paths"]])
query = util.batch_device(query, device=device)

# Run inference
with torch.no_grad():
    predictions = fws_model(support, query)

for i, pred in enumerate(predictions):
    predicted_class = classes[pred]
    predicted_index = pred.item()  # Convert tensor to integer
    print(f"Query {i+1}: Predicted class '{predicted_class}' (index: {predicted_index})")

# Optional: Print support set for reference
print("\nSupport set:")
for i, class_name in enumerate(classes):
    print(f"Index {i}: {class_name}")