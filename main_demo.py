"""Usage: python main_demo.py path/to/avocado.jpg"""
import sys
import torch
from PIL import Image
from src import classical_model as cm
from src import dl_model as dl
from src.dataset import eval_transforms


def run_avocado_demo(image_path, classical_model, dl_model, class_names):
    cp = classical_model.predict(cm.extract_avocado_features(image_path).reshape(1, -1))[0]
    x = eval_transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(dl.DEVICE)
    dl_model.eval().to(dl.DEVICE)
    with torch.no_grad():
        probs = torch.softmax(dl_model(x), 1)[0].cpu()
    print(f"--- Prediction for {image_path} ---")
    print(f"Approach 1 (Classical ML): {class_names[int(cp)]}")
    print(f"Approach 2 (MobileNetV2) : {class_names[int(probs.argmax())]} ({probs.max():.1%})")


if __name__ == "__main__":
    ck = torch.load("models/mobilenetv2.pth", map_location="cpu")
    names = ck["classes"]
    net = dl.build_avocado_classifier(len(names), pretrained=False)
    net.load_state_dict(ck["state_dict"])
    run_avocado_demo(sys.argv[1], cm.load(), net, names)
