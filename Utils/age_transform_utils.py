import os
import torch
from PIL import Image
import cv2
import torch.nn.functional as F


def resize_frames(img_folder, new_folder):
    img_files = os.listdir(img_folder)

    for file in img_files:
        file_name = os.path.join(img_folder, file)
        if os.path.isfile(file_name):
            img = Image.open(file_name)
            img = img.resize((512, 512))
            img.save(os.path.join(new_folder, file[:-4] + '.png'))


def crop_and_resize(img_path):
    img = cv2.imread(img_path)
    img = img[90:430, :]
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(img_path[:-4] + '_cropped.png', img)
    # img.save(img_path[:-4] + '_cropped.png')


def calc_code_diff(no_age_path, age_path):
    no_age_code = torch.load(no_age_path)
    age_code = torch.load(age_path)

    print(torch.sum(torch.abs(no_age_code['code']['shape_code'][0, :100] - age_code['code']['shape_code'][0, :100])))
    print(torch.sum(torch.abs(no_age_code['code']['shape_code'][0, 100:] - age_code['code']['shape_code'][0, 100:])))
    print(torch.sum(torch.abs(no_age_code['code']['appea_code'][0, :100] - age_code['code']['appea_code'][0, :100])))
    print(torch.sum(torch.abs(no_age_code['code']['appea_code'][0, 100:] - age_code['code']['appea_code'][0, 100:])))


def match_probe(probe_path, gallery_folder):
    probe_code = torch.load(probe_path)
    distances = []
    gallery_files = os.listdir(gallery_folder)

    for path in gallery_files:
        gallery_code = torch.load(os.path.join(gallery_folder, path))

        distance = F.l1_loss(probe_code['code']['shape_code'][0, :100], gallery_code['code']['shape_code'][0, :100])
        # distance = F.cosine_embedding_loss(probe_code['code']['shape_code'], gallery_code['code']['shape_code'], torch.tensor([1]).to("cuda:0"))

        distances.append((path, distance))

    print(distances)


# calc_code_diff("test_data/shane/youngest/no_age/no_age_LatentCodes_e_model_Reso64.pth", "test_data/shane/youngest/75/10x_age_results/75_LatentCodes_e_model_Reso64.pth")
resize_frames("/home/gvc/datasets/FGNET/FGNET/images", "/home/gvc/datasets/FGNET_cropped/images")
# crop_and_resize("/home/gvc/headnerf/test_data/dipesh/dipesh.png")
# match_probe("/home/gvc/headnerf/test_data/probe_gallery_test/age_transform_test/probe_code/probe.pth", "/home/gvc/headnerf/test_data/probe_gallery_test/age_transform_test/gallery_codes")