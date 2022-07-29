import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from AgeTools.paths_config import model_paths
from AgeTools.dex_vgg import VGG


class AgingLoss(nn.Module):

    def __init__(self):
        super(AgingLoss, self).__init__()
        self.age_net = VGG()
        ckpt = torch.load(model_paths['age_predictor'], map_location="cpu")['state_dict']
        ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
        self.age_net.load_state_dict(ckpt)
        self.age_net.cuda()
        self.age_net.eval()
        self.min_age = 0
        self.max_age = 100
        self.counter = 0
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]

        return predict_age

    def extract_age(self, input):
        x = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
        predict_age_pb = self.age_net(x)['fc8']
        predicted_age = self.__get_predicted_age(predict_age_pb)

        # if (self.counter+1) % 50 == 0 or self.counter == 0:
        #     print(predicted_age)

        self.counter += 1

        return predicted_age

    def forward(self, input, target_age, calc_loss=True):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)

        # if (self.counter+1) % 50 == 0 or self.counter == 0 or calc_loss is not True:
        #     transform = T.ToPILImage()
        #     img = transform(input[0])
        #     img.show()

        input = (input-self.mean) / self.std
        output_age = self.extract_age(input) / 100

        if calc_loss:
            loss = 10 * F.mse_loss(output_age, target_age)
            return loss
        else:
            return 100 * output_age