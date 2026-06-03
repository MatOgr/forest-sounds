import torch
import torch.nn as nn

from .pipeline import KnowledgeDistillationLoss, ProductionAudioFrontend


# =====================================================================
# PARAMETRIZED KNOWLEDGE DISTILLATION TRAINING
# (RECORDINGS OR SPECTROGRAMS INPUT)
# =====================================================================
def train_kd(
    student: nn.Module,
    teacher: nn.Module,
    dataloader,
    input_type: str = "recordings",
    frontend: "ProductionAudioFrontend | None" = None,
    epochs: int = 10,
    lr: float = 1e-3,
    temperature: float = 3.0,
    alpha: float = 0.4,
    device: str = "cpu",
):
    """
    Distills `teacher` into `student` on raw recordings or precomputed spectrograms.

    input_type:
        "recordings"   -> batch x = (B, samples); `frontend` applied to make log-mel.
                        Requires `frontend` (defaults to a fresh ProductionAudioFrontend).
        "spectrograms" -> batch x = (B, 1, n_mels, time); fed straight to both models.

    dataloader yields (x, targets).
    Teacher runs frozen in eval mode; only student weights update.
    """
    if input_type not in ("recordings", "spectrograms"):
        raise ValueError(
            f"input_type must be 'recordings' or 'spectrograms', got {input_type!r}"
        )

    if input_type == "recordings" and frontend is None:
        frontend = ProductionAudioFrontend()

    student = student.to(device)
    teacher = teacher.to(device)
    if frontend is not None:
        frontend = frontend.to(device)

    criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    teacher.eval()  # Frozen teacher — no grad, no weight updates.
    student.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for x, targets in dataloader:
            x, targets = x.to(device), targets.to(device)

            # Recordings need frontend transform; spectrograms are model-ready.
            if input_type == "recordings":
                x = frontend(x)

            with torch.no_grad():
                teacher_logits = teacher(x)

            optimizer.zero_grad()
            student_logits = student(x)
            loss = criterion(student_logits, teacher_logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg = running_loss / len(dataloader.dataset)
        print(f"[KD/{input_type}] epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    return student


# =====================================================================
# DEMONSTRATION
# =====================================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    from .pipeline import ESCAudioBackbone

    teacher = ESCAudioBackbone(num_classes=50)
    student = ESCAudioBackbone(num_classes=50)

    # Mock recordings: (N, samples)
    x = torch.randn(16, 64_000)
    y = torch.randint(0, 50, (16,))
    dl = DataLoader(TensorDataset(x, y), batch_size=4)

    train_kd(student, teacher, dl, input_type="recordings", epochs=2)
