import torch
import torch.nn.functional as F


def contrastive(logits_per_image, logits_per_text, ground_truth):
    loss_i = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
    loss_t = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
    return (loss_i + loss_t) / 2


def siglip_loss(logits, labels):
    batch_size, n_classes = logits.shape

    targets = torch.full_like(logits, -1.0)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)

    loss = -F.logsigmoid(targets * logits).sum() / batch_size
    return loss