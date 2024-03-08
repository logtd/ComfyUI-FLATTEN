from ..utils.trajectories import sample_trajectories
from ..utils.batching_utils import create_windows_static_standard
import comfy.model_management
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large


class TrajectoryNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "context_length": ("INT", {"default": 20, "min": 0, "max": 999, "step": 1}),
                             "context_overlap": ("INT", {"default": 10, "min": 0, "step": 1}),
                             }}
    RETURN_TYPES = ("TRAJECTORY",)
    FUNCTION = "sample"

    CATEGORY = "flatten"

    def sample(self, images, context_length, context_overlap):
        device = comfy.model_management.get_torch_device()
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights,
                           progress=False).to(device)

        windows = create_windows_static_standard(
            images.shape[0], context_length, context_overlap)
        pbar = comfy.utils.ProgressBar(len(windows))
        trajectory = {
            'trajectory_windows': {},
            'context_windows': windows,
        }
        for i, window in enumerate(windows):
            traj = sample_trajectories(images[window], model, weights, device)
            trajectory['trajectory_windows'][window[0]] = traj
            pbar.update_absolute(i + 1, len(windows))
        return (trajectory,)
