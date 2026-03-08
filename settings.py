import os

MODEL_DATA = "model_data"
if os.getenv("SABINE", False):
    MODEL_DATA = "model_data"
    MODEL_DATA_SIGLIP = "model_data_siglip"
else:
    MODEL_DATA = ""
    MODEL_DATA_SIGLIP = ""

