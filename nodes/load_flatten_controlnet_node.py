import folder_paths
import comfy.controlnet
import comfy.cldm.cldm
from ..modules.flatten_cldm import FlattenControlNet


class FlattenControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name):
        controlnet_path = folder_paths.get_full_path(
            "controlnet", control_net_name)
        original_controlnet = comfy.cldm.cldm.ControlNet
        # Hack
        comfy.cldm.cldm.ControlNet = FlattenControlNet
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        comfy.cldm.cldm.ControlNet = original_controlnet
        return (controlnet,)
