import open_clip
import torch
import torch.nn as nn

class BilinearSigLIP(nn.Module):
    def __init__(self, model_name="ViT-B-16-SigLIP", device="cuda", freeze_backbone=True, upper_triangle=True):
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='webli', device=device
        )

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if hasattr(self.model.visual, 'output_dim'):
            embed_dim = self.model.visual.output_dim
        else:
            embed_dim = self.model.visual.trunk.num_features if hasattr(self.model.visual, 'trunk') else \
            self.model.visual.output_shape[-1]

        self.W = nn.Parameter(torch.eye(embed_dim, device=device))

        self.upper_triangle = upper_triangle
        if self.upper_triangle:
            self.register_buffer('tri_mask', torch.triu(torch.ones(embed_dim, embed_dim)))

    def forward(self, images, text_tokens=None, text_features=None):

        I_f = self.model.encode_image(images)
        T_f = text_features if text_features is not None else self.model.encode_text(text_tokens)

        I_f = I_f / I_f.norm(dim=-1, keepdim=True)
        T_f = T_f / T_f.norm(dim=-1, keepdim=True)

        W_curr = self.W * self.tri_mask if self.upper_triangle else self.W

        logits = (I_f @ W_curr @ T_f.t())

        logits = logits * self.model.logit_scale.exp() + self.model.logit_bias

        return logits