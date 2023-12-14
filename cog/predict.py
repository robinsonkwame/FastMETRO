
from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import transforms

import logging

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

logger = setup_logger()

STATE_DICT_PATH = "./cog/model/PointHMR-HR32-Human3.6M_state_dict.bin"
DEVICE = "cuda"

class Predictor(BasePredictor):
    def setup(self):
        global logger
        logger.info(f"Loading state dict from checkpoint {STATE_DICT_PATH}")
        self.Graphormer_model = torch.load(STATE_DICT_PATH)
        self.Graphormer_model.to(
            DEVICE
        )
        logger.info(f"Done!")

    def run_inference(args, image_list, Graphormer_model):
        self.Graphormer_model.eval()

        with torch.no_grad():
            for image_file in image_list:
                if 'pred' not in image_file:
                    att_all = []
                    img = Image.open(image_file)
                    img_size = img.size
                    # 256x512, img_size = (256,512).
                    x = img_size[0]
                    y = img_size[1]

                    img_tensor = transform(img)
                    img_visual = transform_visualize(img)

                    batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
                    batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
                    # forward-pass
                    outputs = Graphormer_model(batch_imgs)

                    pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, heat_map = outputs

        return {
            'pred_camera': pred_camera,
            'pred_3d_joints': pred_3d_joints,
            'pred_vertices_sub2': pred_vertices_sub2,
            'pred_vertices_sub': pred_vertices_sub,
            'pred_vertices': pred_vertices,
            'heat_map': heat_map
        }


    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(description="Input .jpg image")
    ) -> Dict[str, list]:
        pass
        # """Run a single prediction on the model"""
        # img = Image.open(image) # for easier copypasta

        # # forward-pass
        # out = self.fastmetro(img)

        # # Detach tensors and convert to numpy arrays
        # pred_3d_vertices_intermediate = out['pred_3d_vertices_intermediate'].detach().cpu().numpy()
        # pred_cam = out['pred_cam'].detach().cpu().numpy()

        # # Convert numpy arrays to Python lists
        # pred_3d_vertices_intermediate_list = pred_3d_vertices_intermediate.tolist()
        # pred_cam_list = pred_cam.tolist()

        # # Return as dictionary
        # return {"pred_3d_vertices_intermediate": pred_3d_vertices_intermediate_list, "pred_cam": pred_cam_list}