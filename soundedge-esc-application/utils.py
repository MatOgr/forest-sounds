import torch
import torch.nn.functional as F

from model import CNN_PCAw_SSRPMS_KAN
from classes import ESC50_CLASSES
from compression.pruning import apply_structural_pruning
from compression.qat import prepare_qat_model, convert_qat_model

def load_model(model_path: str, device: torch.device, num_classes: int):
    model = CNN_PCAw_SSRPMS_KAN(num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=device)

    # Case 1: pure state_dict saved with torch.save(model.state_dict(), path)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        raise ValueError("Unsupported checkpoint format.")

    model.to(device)
    model.eval()
    return model

def load_compressed_model(model_path: str, compressed_model_path: str, num_classes: int):
    model = CNN_PCAw_SSRPMS_KAN(num_classes=num_classes)

    original_checkpoint = torch.load(model_path, map_location="cpu")
    compressed_checkpoint = torch.load(compressed_model_path, map_location="cpu")
    try:
        model.load_state_dict(original_checkpoint)
        model = apply_structural_pruning(model)
        model = prepare_qat_model(model)
        model = convert_qat_model(model)
        model.load_state_dict(compressed_checkpoint)
    except RuntimeError as e:
        print(f"Error loading compressed model: {e}")
        raise ValueError("Unsupported checkpoint format.")

    model.to("cpu")
    model.eval()
    return model

@torch.no_grad()
def predict(model, input_tensor: torch.Tensor, device: torch.device):
    input_tensor = input_tensor.to(device)

    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    top_idx = torch.argmax(probs).item()
    top_class = ESC50_CLASSES[top_idx]
    top_prob = probs[top_idx].item()

    all_probs = [
        {"class_name": ESC50_CLASSES[i], "probability": float(probs[i])}
        for i in range(len(ESC50_CLASSES))
    ]

    all_probs = sorted(all_probs, key=lambda x: x["probability"], reverse=True)

    return top_class, top_prob, all_probs