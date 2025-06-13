from doctest import master
import torch
from thop import profile
from models import models as TSN_model

device = torch.device("cpu")
#input_shape of model,batch_size=1
model = TSN_model.TSN(3, 8, 'RGB', base_model='resnet50', is_shift=False)

input = torch.randn(1, 8 * 3, 224, 224)
flops, params = profile(model, inputs=(input, ))

print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
print("params=", str(params/1e6)+'{}'.format("M"))