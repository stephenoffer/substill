"""Channel-narrow and distill a small ResNet on synthetic images (CPU).

The vision arm applies the same activation-subspace idea to convolutions:
``channel_variance_scores`` ranks each Bottleneck's inner channels, then
``build_resnet_student`` narrows them with an absorbed init and
``distill_classifier`` trains the student on the teacher's class logits.

Run:
    python examples/vision_resnet.py
"""

from __future__ import annotations

import torch
from torchvision.models.resnet import Bottleneck, ResNet

from substill.vision import build_resnet_student, channel_variance_scores, distill_classifier


def tiny_resnet(num_classes: int = 10) -> ResNet:
    torch.manual_seed(0)
    return ResNet(Bottleneck, [1, 1, 1, 1], num_classes=num_classes).eval()


def synthetic_loader(n: int, batch: int = 4, num_classes: int = 10) -> list[tuple]:
    torch.manual_seed(1)
    return [(torch.randn(batch, 3, 32, 32), torch.randint(0, num_classes, (batch,)))
            for _ in range(n)]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    teacher = tiny_resnet()
    scores = channel_variance_scores(teacher, synthetic_loader(4), n_batches=2)
    student, info = build_resnet_student(teacher, scores, width_ratio=0.5)

    out = distill_classifier(
        teacher, student, synthetic_loader(6), total_steps=6, lr=1e-3,
        generative_kd="forward_kl", val_loader=synthetic_loader(2), log_every=0,
    )

    print(f"teacher parameters: {count_params(teacher):>10,}")
    print(f"student parameters: {count_params(student):>10,}")
    print(f"bottlenecks narrowed: {sum(v['s_width'] < v['width'] for v in info.values())}"
          f"/{len(info)}")
    print(f"student top-1 after {len(out['history'])} steps: {out['student_top1']:.3f}")


if __name__ == "__main__":
    main()
