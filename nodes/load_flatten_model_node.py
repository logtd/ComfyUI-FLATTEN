import comfy.sd
import comfy.model_base
import folder_paths
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
from ..modules.unet import UNetModel as FlattenModel


class FlattenCheckpointLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        original_UNetModel = openaimodel.UNetModel
        openaimodel.UNetModel = FlattenModel
        comfy.model_base.UNetModel = FlattenModel
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        openaimodel.UNetModel = original_UNetModel
        comfy.model_base.UNetModel = original_UNetModel
        return out[:3]
