import clip
import torch
import torch.nn as nn

class BilinearCLIP(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda", freeze_clip=True, upper_triangle=False):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)

        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False

        embed_dim = self.model.visual.output_dim

        model_dtype = self.model.dtype
        self.W = nn.Parameter(torch.eye(embed_dim, device=device, dtype=model_dtype))

        self.upper_triangle = upper_triangle
        if self.upper_triangle:
            print("initializing upper triangle")
            self.register_buffer('tri_mask', torch.triu(torch.ones(embed_dim, embed_dim)))

    def forward(self, images, text_tokens=None, text_features=None):

        assert not (text_features is None and text_tokens is None), "Must provide text_features or text_tokens"

        I_f = self.model.encode_image(images)
        if text_tokens is not None:
            T_f = self.model.encode_text(text_tokens)
        else:
            T_f = text_features

        I_f = I_f / I_f.norm(dim=-1, keepdim=True)
        T_f = T_f / T_f.norm(dim=-1, keepdim=True)

        if self.upper_triangle:
            W_upper = self.W * self.tri_mask
        else:
            self.W

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = (I_f @ W_upper @ T_f.t()) * logit_scale

        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

