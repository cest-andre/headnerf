from os.path import join
import os
import torch
import numpy as np
from NetWorks.HeadNeRFNet import HeadNeRFNet
import cv2
from HeadNeRFOptions import BaseOptions
from Utils.HeadNeRFLossUtils import HeadNeRFLossUtils
from Utils.RenderUtils import RenderUtils
import pickle as pkl
import time
from glob import glob
from tqdm import tqdm
import imageio
import random
import argparse
from tool_funcs import put_text_alignmentcenter


class FittingImage(object):

    def __init__(self, model_path, save_root, gpu_id) -> None:
        super().__init__()
        self.model_path = model_path

        self.device = torch.device("cuda:%d" % gpu_id)
        self.save_root = save_root
        self.opt_cam = True
        self.view_num = 45
        self.duration = 3.0 / self.view_num
        self.model_name = os.path.basename(model_path)[:-4]

        self.build_info()
        self.build_tool_funcs()

    def build_info(self):

        check_dict = torch.load(self.model_path, map_location=torch.device("cpu"))

        para_dict = check_dict["para"]
        self.opt = BaseOptions(para_dict)

        self.featmap_size = self.opt.featmap_size
        self.pred_img_size = self.opt.pred_img_size

        # if not os.path.exists(self.save_root): os.mkdir(self.save_root)

        net = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False)
        net.load_state_dict(check_dict["net"])

        self.net = net.to(self.device)
        self.net.eval()

    def build_tool_funcs(self):
        self.loss_utils = HeadNeRFLossUtils(device=self.device)
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)

        self.xy = self.render_utils.ray_xy
        self.uv = self.render_utils.ray_uv

    def load_data(self, img_path, mask_path, para_3dmm_path):

        # process imgs
        img_size = (self.pred_img_size, self.pred_img_size)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        gt_img_size = img.shape[0]
        if gt_img_size != self.pred_img_size:
            img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        img[mask_img < 0.5] = 1.0

        self.img_tensor = (torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        self.mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)

        # load init codes from the results generated by solving 3DMM rendering opt.
        with open(para_3dmm_path, "rb") as f:
            nl3dmm_para_dict = pkl.load(f)
        base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0).to(self.device)

        self.base_iden = base_code[:, :self.opt.iden_code_dims]
        self.base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        self.base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims
                                                                                        + self.opt.expr_code_dims + self.opt.text_code_dims]
        self.base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]

        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach().unsqueeze(0)
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach().unsqueeze(0)
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        temp_inmat[:, :2, :] *= (self.featmap_size / gt_img_size)

        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0

        self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device)
        }

    @staticmethod
    def eulurangle2Rmat(angles):
        batch_size = angles.size(0)

        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXs = torch.eye(3, device=angles.device).view(1, 3, 3).repeat(batch_size, 1, 1)
        rotYs = rotXs.clone()
        rotZs = rotXs.clone()

        rotXs[:, 1, 1] = cosx
        rotXs[:, 1, 2] = -sinx
        rotXs[:, 2, 1] = sinx
        rotXs[:, 2, 2] = cosx

        rotYs[:, 0, 0] = cosy
        rotYs[:, 0, 2] = siny
        rotYs[:, 2, 0] = -siny
        rotYs[:, 2, 2] = cosy

        rotZs[:, 0, 0] = cosz
        rotZs[:, 0, 1] = -sinz
        rotZs[:, 1, 0] = sinz
        rotZs[:, 1, 1] = cosz

        res = rotZs.bmm(rotYs.bmm(rotXs))
        return res

    def build_code_and_cam(self):

        # code
        shape_code = torch.cat([self.base_iden + self.iden_offset, self.base_expr + self.expr_offset], dim=-1)
        appea_code = torch.cat([self.base_text, self.base_illu], dim=-1) + self.appea_offset

        opt_code_dict = {
            "bg": None,
            "iden": self.iden_offset,
            "expr": self.expr_offset,
            "appea": self.appea_offset
        }

        code_info = {
            "bg_code": None,
            "shape_code": shape_code,
            "appea_code": appea_code,
        }

        # cam
        if self.opt_cam:
            delta_cam_info = {
                "delta_eulur": self.delta_EulurAngles,
                "delta_tvec": self.delta_Tvecs
            }

            batch_delta_Rmats = self.eulurangle2Rmat(self.delta_EulurAngles)
            base_Rmats = self.cam_info["batch_Rmats"]
            base_Tvecs = self.cam_info["batch_Tvecs"]

            cur_Rmats = batch_delta_Rmats.bmm(base_Rmats)
            cur_Tvecs = batch_delta_Rmats.bmm(base_Tvecs) + self.delta_Tvecs

            batch_inv_inmat = self.cam_info["batch_inv_inmats"]  # [N, 3, 3]
            batch_cam_info = {
                "batch_Rmats": cur_Rmats,
                "batch_Tvecs": cur_Tvecs,
                "batch_inv_inmats": batch_inv_inmat
            }

        else:
            delta_cam_info = None
            batch_cam_info = self.cam_info

        return code_info, opt_code_dict, batch_cam_info, delta_cam_info

    @staticmethod
    def enable_gradient(tensor_list):
        for ele in tensor_list:
            ele.requires_grad = True

    def perform_fitting(self):
        self.delta_EulurAngles = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.delta_Tvecs = torch.zeros((1, 3, 1), dtype=torch.float32).to(self.device)

        self.iden_offset = torch.zeros((1, 100), dtype=torch.float32).to(self.device)
        self.expr_offset = torch.zeros((1, 79), dtype=torch.float32).to(self.device)
        self.appea_offset = torch.zeros((1, 127), dtype=torch.float32).to(self.device)

        if self.opt_cam:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset, self.delta_EulurAngles, self.delta_Tvecs]
            )
        else:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset]
            )

        init_learn_rate = 0.01

        step_decay = 300
        iter_num = 300

        params_group = [
            {'params': [self.iden_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.expr_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.appea_offset], 'lr': init_learn_rate * 1.0},
        ]

        if self.opt_cam:
            params_group += [
                {'params': [self.delta_EulurAngles], 'lr': init_learn_rate * 0.1},
                {'params': [self.delta_Tvecs], 'lr': init_learn_rate * 0.1},
            ]

        optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
        lr_func = lambda epoch: 0.1 ** (epoch / step_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

        gt_img = (self.img_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        loop_bar = tqdm(range(iter_num), leave=True)
        for iter_ in loop_bar:
            with torch.set_grad_enabled(True):
                code_info, opt_code_dict, cam_info, delta_cam_info = self.build_code_and_cam()

                pred_dict = self.net("test", self.xy, self.uv, **code_info, **cam_info)

                batch_loss_dict = self.loss_utils.calc_total_loss(
                    delta_cam_info=delta_cam_info, opt_code_dict=opt_code_dict, pred_dict=pred_dict,
                    gt_rgb=self.img_tensor, mask_tensor=self.mask_tensor
                )

            optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            optimizer.step()
            scheduler.step()
            loop_bar.set_description(
                f'Opt, Head Loss: {batch_loss_dict["head_loss"].item():.6f}, Total Loss: {batch_loss_dict["total_loss"].item():.6f}  ')

            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # cv2.imwrite("./temp_res/opt_imgs/img_%04d.png" % iter_, coarse_fg_rgb[:, :, ::-1])

        #   Obtain cam info from build function, but pass age transformed code info to net for image generation.
        # code_info, opt_code_dict, cam_info, delta_cam_info = self.build_code_and_cam()
        # exemplar_code = torch.load("/home/gvc/headnerf/test_data/TED_video/exemplar/75/10x_age_results/LatentCodes_plar_model_Reso64.pth")['code']

        #   First hundred parameters in shape is identity.  Leave expression parameters unchanged.  Likewise with appearance (change texture, not illumination).
        # code_info['shape_code'][0, :100] = exemplar_code['shape_code'][0, :100]
        # code_info['appea_code'][0, :100] = exemplar_code['appea_code'][0, :100]
        # pred_dict = self.net("test", self.xy, self.uv, **code_info, **cam_info)

        coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
        coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)

        self.res_img = res_img
        self.res_code_info = code_info
        self.res_cam_info = cam_info

    def save_res(self, base_name, save_root):

        # Generate Novel Views
        render_nv_res = self.render_utils.render_novel_views(self.net, self.res_code_info)
        NVRes_save_path = "%s/FittingResNovelView_%s.gif" % (save_root, base_name)
        imageio.mimsave(NVRes_save_path, render_nv_res, 'GIF', duration=self.duration)

        # Generate Rendered FittingRes
        img_save_path = "%s/FittingRes_%s.png" % (save_root, base_name)

        #   Render video frame.
        # solo_img_save_path = "%s/SoloRes_%s.png" % ('/home/gvc/headnerf/test_data/TED_video/all_transformed_frames/75/10x_age_results', base_name)
        # cv2.imwrite(solo_img_save_path, self.res_img[:, :, ::-1])

        # self.res_img = put_text_alignmentcenter(self.res_img, self.pred_img_size, "Input", (0,0,0), offset_x=0)
        # self.res_img = put_text_alignmentcenter(self.res_img, self.pred_img_size, "Fitting", (0,0,0), offset_x=self.pred_img_size,)

        # self.res_img = cv2.putText(self.res_img, "Input", (110, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        # self.res_img = cv2.putText(self.res_img, "Fitting", (360, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        cv2.imwrite(img_save_path, self.res_img[:, :, ::-1])

        if self.tar_code_info is not None:
            # "Generate Morphing Res"
            morph_res = self.render_utils.render_morphing_res(self.net, self.res_code_info, self.tar_code_info,
                                                              self.view_num)
            morph_save_path = "%s/FittingResMorphing_%s.gif" % (save_root, base_name)
            imageio.mimsave(morph_save_path, morph_res, 'GIF', duration=self.duration)

        for k, v in self.res_code_info.items():
            if isinstance(v, torch.Tensor):
                self.res_code_info[k] = v.detach()

        temp_dict = {
            "code": self.res_code_info
        }

        torch.save(temp_dict, "%s/LatentCodes_%s_%s.pth" % (save_root, base_name, self.model_name))

    def fitting_single_images(self, img_path, mask_path, para_3dmm_path, tar_code_path, save_root):

        self.load_data(img_path, mask_path, para_3dmm_path)
        base_name = os.path.basename(img_path)[:-4]
        # base_name = os.path.basename(img_path)[4:-4]

        # load tar code
        if tar_code_path is not None:
            temp_dict = torch.load(tar_code_path, map_location="cpu")
            tar_code_info = temp_dict["code"]
            for k, v in tar_code_info.items():
                if v is not None:
                    tar_code_info[k] = v.to(self.device)
            self.tar_code_info = tar_code_info
        else:
            self.tar_code_info = None

        self.perform_fitting()
        self.save_res(base_name, save_root)


if __name__ == "__main__":

    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    # torch.backends.cudnn.deterministic = True  # cudnn
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(description='a framework for fitting a single image using HeadNeRF')
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--para_3dmm", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)

    parser.add_argument("--target_embedding", type=str, default="")

    args = parser.parse_args()

    model_path = args.model_path
    save_root = args.save_root

    img_path = args.img
    mask_path = args.mask
    para_3dmm_path = args.para_3dmm

    if len(args.target_embedding) == 0:
        target_embedding_path = None
    else:
        target_embedding_path = args.target_embedding

        temp_str_list = target_embedding_path.split("/")
        if temp_str_list[1] == "*":
            temp_str_list[1] = os.path.basename(model_path)[:-4]
            target_embedding_path = os.path.join(*temp_str_list)

        assert os.path.exists(target_embedding_path)

    tt = FittingImage(model_path, save_root, gpu_id=0)

    for path in os.listdir(img_path):
        img_name = path[:-4]
        img_data_root = os.path.join(mask_path, img_name)
        temp_img_path = os.path.join(img_path, path)
        temp_mask_path = img_data_root + "/" + img_name + "_mask.png"
        temp_param_path = img_data_root + "/" + img_name + "_1x_img_10x_id_l1_loss_nl3dmm.pkl"
        temp_save_root = os.path.join(img_data_root, save_root)

        if os.path.exists(temp_save_root) is False:
            os.mkdir(temp_save_root)

        # tt.save_root = temp_save_root
        tt.fitting_single_images(temp_img_path, temp_mask_path, temp_param_path, target_embedding_path, temp_save_root)
