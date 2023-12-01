import argparse
import torch
from PIL import Image
from torchvision import transforms
from utils.modeling_xyz_fastmetro import FastMETRO_Body_Network as FastMETRO_Network
from utils.hrnet_cls_net_featmaps import get_cls_net
from utils.default import update_config as hrnet_update_config
from utils.default import _C as hrnet_config
import logging

import numpy as np
from typing import Dict
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_visualize = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor()])    

    def parse_default_args(self):
        parser = argparse.ArgumentParser()
        #########################################################
        # Data related arguments
        #########################################################
        parser.add_argument("--image_file_or_path", default='./test_images/human-body', type=str, 
                            help="test data")
        #########################################################
        # Loading/Saving checkpoints
        #########################################################
        parser.add_argument("--resume_checkpoint", default="models/FastMETRO-L-H64_h36m_state_dict.bin", type=str, required=False,
                            help="Path to specific checkpoint for resume training.")
        #########################################################
        # Model architectures
        #########################################################
        parser.add_argument("--model_name", default='FastMETRO-L', type=str,
                            help='Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L')
        parser.add_argument("--model_dim_1", default=512, type=int)
        parser.add_argument("--model_dim_2", default=128, type=int)
        parser.add_argument("--feedforward_dim_1", default=2048, type=int)
        parser.add_argument("--feedforward_dim_2", default=512, type=int)
        parser.add_argument("--conv_1x1_dim", default=2048, type=int)
        parser.add_argument("--transformer_dropout", default=0.1, type=float)
        parser.add_argument("--transformer_nhead", default=8, type=int)
        parser.add_argument("--pos_type", default='sine', type=str)    
        parser.add_argument("--use_smpl_param_regressor", default=False, action='store_true',) 
        # CNN backbone
        parser.add_argument('-a', '--arch', default='hrnet-w64',
                            help='CNN backbone architecture: hrnet-w64, resnet50')
        #########################################################
        # Others
        #########################################################
        parser.add_argument("--device", type=str, default='cuda', 
                            help="cuda or cpu")
        parser.add_argument('--seed', type=int, default=88, 
                            help="random seed for initialization.")

        return parser.parse_args()

    def setup(self):
        # So this setup is cobbled together from a Dockerized version that I made
        # to prove that FastMETRO could be containerized and independent of SMPL

        # This means there's a lot of cruft here that is difficult to parse. But
        # I do try to organize it into steps

        # Step 1: Define default arguments
        self.args = self.parse_default_args()
        args = self.args # for easier copypasta of code
        
        # Step 2: Attach GPUS, load FastMETRO network
        args.num_gpus = torch.cuda.device_count()
        args.distributed = args.num_gpus > 1
        args.device = torch.device(args.device)

        # Load pretrained model    
        logging.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))
        if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None') and ('state_dict' not in args.resume_checkpoint):
            # if only run eval, load checkpoint
            logging.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
            _FastMETRO_Network = torch.load(args.resume_checkpoint)
            logging.info("(... must work w pred 3d joints and 3d vertices)".format(args.resume_checkpoint))
        else:
            # init ImageNet pre-trained backbone model
            if args.arch == 'hrnet-w64':
                hrnet_yaml = 'models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                hrnet_checkpoint = 'models/hrnetv2_w64_imagenet_pretrained.pth'
                hrnet_update_config(hrnet_config, hrnet_yaml)
                backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
                logging.info('=> loading hrnet-v2-w64 model')
            else:
                assert False, "The CNN backbone name is not valid"

            _FastMETRO_Network = FastMETRO_Network(args, backbone)
            # number of parameters
            overall_params = sum(p.numel() for p in _FastMETRO_Network.parameters() if p.requires_grad)
            backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            transformer_params = overall_params - backbone_params
            logging.info('Number of CNN Backbone learnable parameters: {}'.format(backbone_params))
            logging.info('Number of Transformer Encoder-Decoder learnable parameters: {}'.format(transformer_params))
            logging.info('Number of Overall learnable parameters: {}'.format(overall_params))

            if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None'):
                # for fine-tuning or resume training or inference, load weights from checkpoint
                logging.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
                cpu_device = torch.device('cpu')
                state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
                _FastMETRO_Network.load_state_dict(state_dict, strict=False)
                del state_dict

        self.fastmetro = _FastMETRO_Network.to(args.device)

    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(description="Input .jpg image")
    ) -> Dict[str, list]:
        """Run a single prediction on the model"""
        img = Image.open(image) # for easier copypasta
        img_tensor = self.transform(img)
        img_visual = self.transform_visualize(img)

        batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
        batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
        
        # forward-pass
        out = self.fastmetro(batch_imgs)

        logging.warn(f"{str(out['pred_3d_vertices_coarse'].shape)}")

        # Detach tensors and convert to numpy arrays
        pred_3d_vertices_intermediate = out['pred_3d_vertices_intermediate'].detach().cpu().numpy()
        pred_cam = out['pred_cam'].detach().cpu().numpy()

        # Convert numpy arrays to Python lists
        pred_3d_vertices_intermediate_list = pred_3d_vertices_intermediate.tolist()
        pred_cam_list = pred_cam.tolist()

        # Return as dictionary
        return {"pred_3d_vertices_intermediate": pred_3d_vertices_intermediate_list, "pred_cam": pred_cam_list}