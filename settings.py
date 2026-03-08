import os

MODEL_DATA = "model_data"
if os.getenv("SABINE", False):
    MODEL_DATA = "model_data"
    MODEL_DATA_SIGLIP = "model_data_siglip"
else:
    MODEL_DATA = "/media/pmantini/New Volume/Research/Bilinear/model_data_contrastive"
    MODEL_DATA_SIGLIP = "/media/pmantini/New Volume/Research/Bilinear/siglip"

