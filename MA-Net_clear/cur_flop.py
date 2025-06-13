import torch
from thop import profile
from models import models as TSN_model

device = torch.device("cpu")
#input_shape of model,batch_size=1
# model = TSN_model.TSN(3, 30, 'RGB', base_model='resnet50', is_shift=True, new_length=1, img_feature_dim=224)  # FLOPs= 124.27971732G  params= 25.692883M
# model = TSN_model.TSN(3, 30, 'RGB', base_model='resnet50', module='CS', img_feature_dim=224)    # FLOPs= 124.27971732G, params= 25.692883M
# model = TSN_model.TSN(3, 30, 'RGB', base_model='resnet50', module='KF2_CS_2', img_feature_dim=224)  # FLOPs= 124.279733144G, pytorchparams= 25.707283M
# model = TSN_model.TSN(3, 30, 'RGB', base_model='resnet50', module='KF2_CS_2_1', img_feature_dim=224) # FLOPs= 124.279733144G  params= 25.707283M
# model = TSN_model.TSN(3, 30, 'RGB', base_model='resnet50', module='KF2_CS_4_1', img_feature_dim=224)    # FLOPs= 124.279733144G  params= 25.707283M
# model = TSN_model.TSN(3, 30, 'RGB', base_model='resnet50', module='res2_4', img_feature_dim=224) #FLOPs= 124.118089283G params= 24.927577M
model = TSN_model.TSN(3, 30, 'RGB', base_model='resnet50', img_feature_dim=224) # resnet50 FLOPs= 123.95102208G params= 23.514179M
input = torch.randn(1, 30 * 3, 224, 224)
flops, params = profile(model, inputs=(input, ))

print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
print("params=", str(params/1e6)+'{}'.format("M"))