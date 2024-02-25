from ..utils.trajectories import sample_trajectories
import comfy.model_management


class TrajectoryNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             }}
    RETURN_TYPES = ("TRAJECTORY",)
    FUNCTION = "sample"

    CATEGORY = "flatten"

    def sample(self, images):
        return (sample_trajectories(images, comfy.model_management.get_torch_device()),)
