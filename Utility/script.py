
import torch, sys
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
try:
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        print("props:", torch.cuda.get_device_properties(0))
except Exception as e:
    print("CUDA property error:", e)