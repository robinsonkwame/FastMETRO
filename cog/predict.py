from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        self.metro = None

    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(description="Input .jpg image")
    ) -> Dict[str, list]:
        """Run a single prediction on the model"""
        img = Image.open(image) # for easier copypasta

        # forward-pass
        out = self.fastmetro(img)

        # Detach tensors and convert to numpy arrays
        pred_3d_vertices_intermediate = out['pred_3d_vertices_intermediate'].detach().cpu().numpy()
        pred_cam = out['pred_cam'].detach().cpu().numpy()

        # Convert numpy arrays to Python lists
        pred_3d_vertices_intermediate_list = pred_3d_vertices_intermediate.tolist()
        pred_cam_list = pred_cam.tolist()

        # Return as dictionary
        return {"pred_3d_vertices_intermediate": pred_3d_vertices_intermediate_list, "pred_cam": pred_cam_list}