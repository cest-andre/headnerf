import torch
from torch import nn
import torch.nn.functional as F
from AgeTools.paths_config import model_paths
from AgeTools.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.cuda()
        self.facenet.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.counter = 0

    def extract_feats(self, x):
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, rend_img, gt_img):
        rend_img = (rend_img - self.mean) / self.std
        gt_img = (gt_img - self.mean) / self.std

        rend_feats = self.extract_feats(rend_img)
        gt_feats = self.extract_feats(gt_img)

        loss = 50 * F.l1_loss(rend_feats, gt_feats)
        # loss = 10 * F.cosine_embedding_loss(rend_feats, gt_feats, torch.tensor([1]).to("cuda:0"))

        # if (self.counter+1) % 50 == 0 or self.counter == 0:
        #     print(loss)

        self.counter += 1

        return loss