import torch
from einops import rearrange
from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation
import comfy.sd
import comfy.model_base
import folder_paths
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
from ..modules.unet import UNetModel as FlattenModel


class PatchBaseModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, *args, model_type=comfy.model_base.ModelType.EPS, device=None, unet_model=FlattenModel, **kwargs):
        super().__init__(model_config, model_type, device, FlattenModel)


class PatchSDXL(PatchBaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = openaimodel.Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(
            **{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = comfy.model_base.sdxl_pooled(
            kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(
            dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class FlattenCheckpointLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        original_base = comfy.model_base.BaseModel
        original_sdxl = comfy.model_base.SDXL
        comfy.model_base.BaseModel = PatchBaseModel
        comfy.model_base.SDXL = PatchSDXL
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        comfy.model_base.BaseModel = original_base
        comfy.model_base.SDXL = original_sdxl

        def model_function_wrapper(apply_model_func, apply_params):
            # Prepare 3D latent
            input_x = apply_params['input']
            len_conds = len(apply_params['cond_or_uncond'])
            frame_count = input_x.shape[0] // len_conds
            input_x = rearrange(
                input_x, "(b f) c h w -> b c f h w", b=len_conds)
            timestep_ = apply_params['timestep']
            timestep_ = timestep_[torch.arange(
                0, timestep_.size(0), frame_count)]

            # Correct Flatten vars for any batching

            # Do injection if needed
            transformer_options = apply_params['c'].get(
                'transformer_options', {})
            flatten_options = transformer_options.get('flatten', {})

            idxs = None
            context_start = 0
            if 'ad_params' in transformer_options and transformer_options['ad_params'].get('sub_idxs', None) is not None:
                idxs = transformer_options['ad_params']['sub_idxs']
                context_start = idxs[0]
            else:
                idxs = list(range(frame_count))
                transformer_options['flatten']['trajs'] = transformer_options['flatten']['trajs_windows'][0]
            transformer_options['flatten']['trajs'] = transformer_options['flatten']['trajs_windows'][context_start]

            transformer_options['flatten']['idxs'] = idxs
            transformer_options['flatten']['video_length'] = frame_count

            # Inject if sampling
            injection_handler = flatten_options.get('injection_handler', None)
            if injection_handler is not None:
                step = flatten_options['injection_handler'](
                    timestep_[0], context_start, len_conds)
                flatten_options['step'] = step

            del apply_params['timestep']
            conditioning = {}
            for key in apply_params['c']:
                value = apply_params['c'][key]
                if key == 'c_crossattn':
                    value = value[torch.arange(0, value.size(0), frame_count)]

                conditioning[key] = value

            conditioning
            del apply_params['c']
            del apply_params['input']
            model_out = apply_model_func(input_x, timestep_, **conditioning)

            # Save injections if unsampling
            save_injections_handler = flatten_options.get(
                'save_injections_handler', None)
            if save_injections_handler is not None:
                save_injections_handler(context_start)

            # Return 2D latent
            model_out = rearrange(model_out, 'b c f h w -> (b f) c h w')
            return model_out

        model = out[0]
        model.model_options['model_function_wrapper'] = model_function_wrapper

        return out[:3]
