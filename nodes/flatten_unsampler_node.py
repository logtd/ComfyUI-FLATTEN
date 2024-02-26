from einops import rearrange
import comfy.samplers
import torch


class UnsamplerFlattenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "normalize": (["disable", "enable"], ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "trajectories": ("TRAJECTORY", ),
                 "old_qk": ("INT", {"default": 0, "min": 0, "max": 1}),
                 }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "unsampler"

    CATEGORY = "sampling"

    def unsampler(self, model, cfg, sampler_name, steps, end_at_step, scheduler, normalize, positive, negative, latent_image, trajectories, old_qk):
        # PREPARTION
        latent = latent_image
        latent_image = latent["samples"]
        # latent_image = rearrange(latent_image, "(b f) c h w -> b c f h w", b=1)
        transformer_options = {}
        if 'transformer_options' in model.model_options:
            transformer_options = model.model_options['transformer_options']

        transformer_options = {
            **transformer_options,
            'flatten': {
                'trajs': trajectories,
                'old_qk': old_qk
            }
        }
        model.model_options['transformer_options'] = transformer_options

        normalize = False
        device = comfy.model_management.get_torch_device()

        end_at_step = min(end_at_step, steps-1)
        end_at_step = steps - end_at_step

        noise = torch.zeros(latent_image.size(
        ), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(
                latent["noise_mask"], noise.shape, device)

        real_model = None
        real_model = model.model

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        positive = comfy.sample.convert_cond(positive)
        negative = comfy.sample.convert_cond(negative)

        models, inference_memory = comfy.sample.get_additional_models(
            positive, negative, model.model_dtype())

        comfy.model_management.load_models_gpu(
            [model] + models, model.memory_required(noise.shape) + inference_memory)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sigmas = sampler.sigmas.flip(0) + 0.0001
        injection_dict = self._get_blank_injection_dict()
        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)
            self._update_injections(model, injection_dict)

        self._clear_injections(model)

        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False,
                                 denoise_mask=noise_mask, sigmas=sigmas, start_step=0, last_step=end_at_step, callback=callback)

        self._clear_injections(model)
        if normalize:
            # technically doesn't normalize because unsampling is not guaranteed to end at a std given by the schedule
            samples -= samples.mean()
            samples /= samples.std()
        samples = samples.cpu()

        comfy.sample.cleanup_additional_models(models)

        return ({'samples': samples, 'injections': injection_dict},)

    def _get_blank_injection_dict(self):
        return {
            'features0': [],
            'features1': [],
            'features2': [],
            'q4': [],
            'k4': [],
            'q5': [],
            'k5': [],
            'q6': [],
            'k6': [],
            'q7': [],
            'k7': [],
            'q8': [],
            'k8': [],
            'q9': [],
            'k9': []
        }

    def _clear_injections(self, model):
        model = model.model.diffusion_model
        res_attn_dict = {1: [0, 1], 2: [0]}
        for res in res_attn_dict:
            for block in res_attn_dict[res]:
                model.output_blocks[3*res+block][0].out_layers_features = None
        attn_res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0]}
        for attn in attn_res_dict:
            for block in attn_res_dict[attn]:
                module = model.output_blocks[3*attn +
                                             block][1].transformer_blocks[0].attn1
                module.q = None
                module.k = None
                module.inject_q = None
                module.inject_k = None

    def _update_injections(self, model, injection):
        model = model.model.diffusion_model

        res_dict = {1: [0, 1], 2: [0]}
        res_idx = 0
        for res in res_dict:
            for block in res_dict[res]:
                injection[f'features{res_idx}'].append(
                    model.output_blocks[3*res+block][0].out_layers_features.cpu())
                res_idx += 1

        attn_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0]}
        attn_idx = 4
        for attn in attn_dict:
            for block in attn_dict[attn]:
                module = model.output_blocks[3*attn +
                                             block][1].transformer_blocks[0].attn1
                injection[f'q{attn_idx}'].append(module.q.cpu())
                injection[f'k{attn_idx}'].append(module.k.cpu())
                attn_idx += 1
