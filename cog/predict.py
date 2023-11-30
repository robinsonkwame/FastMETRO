# Stock predict to gradually leverage the setup 

# first we set up cog/Docker
#   x base image, no libraries
#   x with requirements (note had to set to python 3.8)
#   
# then we verify imports, gradually set up the network
# then we add the network to the predict method

# base imports
import argparse
import torch
from PIL import Image
from torchvision import transforms
from utils.modeling_xyz_fastmetro import FastMETRO_Body_Network as FastMETRO_Network
from utils.hrnet_cls_net_featmaps import get_cls_net
from utils.default import update_config as hrnet_update_config
from utils.default import _C as hrnet_config
import numpy as np
import os
import sys
import logging
from logging import StreamHandler, Handler, getLevelName


from cog import BasePredictor, Input, Path
import torch
import logging

class Predictor(BasePredictor):
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_visualize = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor()])    
    def setup(self):
        # So this setup is cobbled together from a Dockerized version that I made
        # to prove that FastMETRO could be containerized and independent of SMPL

        # This means there's a lot of cruft here that is difficult to parse. But
        # I do try to organize it into steps

        # Step 1: Define default arguments

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

        self.args = parser.parse_args()
        #self.model = torch.load("./weights.pth")

    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(description="Grayscale input image")
    ) -> str:
        """Run a single prediction on the model"""
        logging.warn(f"{str(self.args)}")
        return torch.__version__

# # ---------------------------------------------------------------
# # Inference script to serve vertices from MIT licensed checkpoint
# # that only serves X,Y,Z points while not refering to SMPL derived 
# # adjusments or data.
# # ---------------------------------------------------------------
# import argparse
# import torch
# from PIL import Image
# from torchvision import transforms
# from images.utils.modeling_xyz_fastmetro import FastMETRO_Body_Network as FastMETRO_Network
# from images.utils.hrnet_cls_net_featmaps import get_cls_net
# from images.utils.default import update_config as hrnet_update_config
# from images.utils.default import _C as hrnet_config
# import numpy as np
# import os
# import sys
# import logging
# from logging import StreamHandler, Handler, getLevelName

# transform = transforms.Compose([transforms.Resize(224),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# transform_visualize = transforms.Compose([transforms.Resize(224),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor()])

# def mkdir(path):
#     # if it is the current folder, skip.
#     # otherwise the original code will raise FileNotFoundError
#     if path == '':
#         return
#     try:
#         os.makedirs(path)
#     except OSError as e:
#         if e.errno != e.errno.EEXIST:
#             raise


# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# #
# # this class is a copy of logging.FileHandler except we end self.close()
# # at the end of each emit. While closing file and reopening file after each
# # write is not efficient, it allows us to see partial logs when writing to
# # fused Azure blobs, which is very convenient
# class FileHandler(StreamHandler):
#     """
#     A handler class which writes formatted logging records to disk files.
#     """
#     def __init__(self, filename, mode='a', encoding=None, delay=False):
#         """
#         Open the specified file and use it as the stream for logging.
#         """
#         # Issue #27493: add support for Path objects to be passed in
#         filename = os.fspath(filename)
#         #keep the absolute path, otherwise derived classes which use this
#         #may come a cropper when the current directory changes
#         self.baseFilename = os.path.abspath(filename)
#         self.mode = mode
#         self.encoding = encoding
#         self.delay = delay
#         if delay:
#             #We don't open the stream, but we still need to call the
#             #Handler constructor to set level, formatter, lock etc.
#             Handler.__init__(self)
#             self.stream = None
#         else:
#             StreamHandler.__init__(self, self._open())

#     def close(self):
#         """
#         Closes the stream.
#         """
#         self.acquire()
#         try:
#             try:
#                 if self.stream:
#                     try:
#                         self.flush()
#                     finally:
#                         stream = self.stream
#                         self.stream = None
#                         if hasattr(stream, "close"):
#                             stream.close()
#             finally:
#                 # Issue #19523: call unconditionally to
#                 # prevent a handler leak when delay is set
#                 StreamHandler.close(self)
#         finally:
#             self.release()

#     def _open(self):
#         """
#         Open the current base file with the (original) mode and encoding.
#         Return the resulting stream.
#         """
#         return open(self.baseFilename, self.mode, encoding=self.encoding)

#     def emit(self, record):
#         """
#         Emit a record.

#         If the stream was not opened because 'delay' was specified in the
#         constructor, open it before calling the superclass's emit.
#         """
#         if self.stream is None:
#             self.stream = self._open()
#         StreamHandler.emit(self, record)
#         self.close()

#     def __repr__(self):
#         level = getLevelName(self.level)
#         return '<%s %s (%s)>' % (self.__class__.__name__, self.baseFilename, level)

# def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     # don't log results for the non-master process
#     if distributed_rank > 0:
#         return logger
#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(name)s %(levelname)s: %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     if save_dir:
#         fh = FileHandler(os.path.join(save_dir, filename))
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     return logger


# def parse_args():
#     parser = argparse.ArgumentParser()
#     #########################################################
#     # Data related arguments
#     #########################################################
#     parser.add_argument("--image_file_or_path", default='./test_images/human-body', type=str, 
#                         help="test data")
#     #########################################################
#     # Loading/Saving checkpoints
#     #########################################################
#     parser.add_argument("--output_dir", default='output/', type=str, required=False,
#                         help="The output directory to save checkpoint and test results.")
#     parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
#                         help="Path to specific checkpoint for resume training.")
#     #########################################################
#     # Model architectures
#     #########################################################
#     parser.add_argument("--model_name", default='FastMETRO-L', type=str,
#                         help='Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L')
#     parser.add_argument("--model_dim_1", default=512, type=int)
#     parser.add_argument("--model_dim_2", default=128, type=int)
#     parser.add_argument("--feedforward_dim_1", default=2048, type=int)
#     parser.add_argument("--feedforward_dim_2", default=512, type=int)
#     parser.add_argument("--conv_1x1_dim", default=2048, type=int)
#     parser.add_argument("--transformer_dropout", default=0.1, type=float)
#     parser.add_argument("--transformer_nhead", default=8, type=int)
#     parser.add_argument("--pos_type", default='sine', type=str)    
#     parser.add_argument("--use_smpl_param_regressor", default=False, action='store_true',) 
#     # CNN backbone
#     parser.add_argument('-a', '--arch', default='hrnet-w64',
#                         help='CNN backbone architecture: hrnet-w64, resnet50')
#     #########################################################
#     # Others
#     #########################################################
#     parser.add_argument("--device", type=str, default='cuda', 
#                         help="cuda or cpu")
#     parser.add_argument('--seed', type=int, default=88, 
#                         help="random seed for initialization.")

#     args = parser.parse_args()
#     return args

# def run_inference(args, image_list, FastMETRO_model):
#     # switch to evaluate mode
#     FastMETRO_model.eval()
    
#     for image_file in image_list:
#         if 'pred' not in image_file:
#             img = Image.open(image_file)
#             img_tensor = transform(img)
#             img_visual = transform_visualize(img)

#             batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
#             batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
            
#             # forward-pass
#             out = FastMETRO_model(batch_imgs)
#             # the check points don't provide access to all of the inference information as one variable
#             # so we work with the 3d joints and the coarse vertices seperately

#             # return these two matrices
#             logger.info(
#                 # out['pred_3d_joints'].shape,
#                 # out['pred_3d_vertices_coarse'].shape
#                 "we'd return 3d points in .xyz format"
#             )
    
#     logger.info("The inference completed successfully. Finalizing run...")
#     return


# def main(args):
#     print("FastMETRO for 3D Human Mesh Reconstruction!")
#     global logger
#     # Setup CUDA, GPU & distributed training
#     args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
#     args.distributed = args.num_gpus > 1
#     args.device = torch.device(args.device)

#     mkdir(args.output_dir)
#     logger = setup_logger("FastMETRO Inference", args.output_dir, 0)

#     # Load pretrained model    
#     logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))
#     if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None') and ('state_dict' not in args.resume_checkpoint):
#         # if only run eval, load checkpoint
#         logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
#         _FastMETRO_Network = torch.load(args.resume_checkpoint)
#         logger.info("(... must work w pred 3d joints and 3d vertices)".format(args.resume_checkpoint))
#     else:
#         # init ImageNet pre-trained backbone model
#         if args.arch == 'hrnet-w64':
#             hrnet_yaml = '/FastMETRO/images/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
#             hrnet_checkpoint = '/FastMETRO/images/models/hrnetv2_w64_imagenet_pretrained.pth'
#             hrnet_update_config(hrnet_config, hrnet_yaml)
#             backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
#             logger.info('=> loading hrnet-v2-w64 model')
#         else:
#             assert False, "The CNN backbone name is not valid"

#         _FastMETRO_Network = FastMETRO_Network(args, backbone)
#         # number of parameters
#         overall_params = sum(p.numel() for p in _FastMETRO_Network.parameters() if p.requires_grad)
#         backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
#         transformer_params = overall_params - backbone_params
#         logger.info('Number of CNN Backbone learnable parameters: {}'.format(backbone_params))
#         logger.info('Number of Transformer Encoder-Decoder learnable parameters: {}'.format(transformer_params))
#         logger.info('Number of Overall learnable parameters: {}'.format(overall_params))

#         if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None'):
#             # for fine-tuning or resume training or inference, load weights from checkpoint
#             logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
#             cpu_device = torch.device('cpu')
#             state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
#             _FastMETRO_Network.load_state_dict(state_dict, strict=False)
#             del state_dict

#     _FastMETRO_Network.to(args.device)


#     image_list = []
#     if not args.image_file_or_path:
#         raise ValueError("image_file_or_path not specified")
#     if os.path.isfile(args.image_file_or_path):
#         image_list = [args.image_file_or_path]
#     elif os.path.isdir(args.image_file_or_path):
#         # should be a path with images only
#         for filename in os.listdir(args.image_file_or_path):
#             if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
#                 image_list.append(args.image_file_or_path+'/'+filename) 
#     else:
#         raise ValueError("Cannot find images at {}".format(args.image_file_or_path))
    
#     run_inference(args, image_list, _FastMETRO_Network)

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)