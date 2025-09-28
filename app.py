from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import models, transforms
from PIL import Image
import io, base64, os

app = Flask(__name__, static_folder="web", static_url_path="")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load("models/onepiece_resnet18.pth", map_location=device)
classes = checkpoint["classes"]
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_b64 = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        character = classes[pred.item()]
    return jsonify({"character": character})

@app.route("/", methods=["GET"])
def index():
    return send_from_directory("web", "index.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
