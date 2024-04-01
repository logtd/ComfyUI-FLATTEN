import torch

from ..utils.flow_noise import create_noise_generator


class CreateFlowNoiseNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"latent": ("LATENT",),
                 "trajectories": ("TRAJECTORY",),
                 "add_noise_to_latent": ("BOOLEAN", {"default": False})
                 }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "create"

    CATEGORY = "flatten"

    def create(self, latent, trajectories, add_noise_to_latent):

        noise = torch.zeros_like(latent['samples'])

        noise_gen = create_noise_generator(
            [traj['directions'] for traj in trajectories['trajectory_windows'].values()], noise.shape[0])

        noise = noise_gen(noise)(None, None)

        if add_noise_to_latent:
            noise += latent['samples']
            latent = latent.copy()
            latent['sampels'] = noise
            return (latent, )

        return ({'samples': noise}, )
