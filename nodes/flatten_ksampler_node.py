from einops import rearrange
import comfy.samplers
import torch


class KSamplerFlattenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "add_noise": (["enable", "disable"], ),
                 "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "injection_steps": ("INT", {"default": 15, "min": 1, "max": 10000}),
                 "old_qk": ("INT", {"default": 0, "min": 0, "max": 1}),
                 "trajectories": ("TRAJECTORY",),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                 "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                 "return_with_leftover_noise": (["disable", "enable"], ),
                 }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, injection_steps, old_qk, trajectories, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        # PREPARTION
        injections = latent_image["injections"]

        latent = latent_image
        latent_image = latent["samples"]

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

        device = comfy.model_management.get_torch_device()

        end_at_step = steps

        noise = torch.zeros(latent_image.size(
        ), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(
                latent["noise_mask"], noise.shape, device)

        noise = torch.zeros(latent_image.size(
        ), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None

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

        sigmas = sampler.sigmas

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            self._clear_injections(model)
            if step + 1 < injection_steps:
                self._inject(model, injections, device, step + 1)
            pbar.update_absolute(step + 1, total_steps)

        self._clear_injections(model)
        if start_at_step < injection_steps:
            self._inject(model, injections, device, start_at_step)

        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False,
                                 denoise_mask=noise_mask, sigmas=sigmas, start_step=start_at_step, last_step=end_at_step, callback=callback)
        self._clear_injections(model)
        samples = samples.cpu()

        comfy.sample.cleanup_additional_models(models)

        out = {}
        out["samples"] = samples

        return (out, )

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

    def _inject(self, model, injection, device, step):
        model = model.model.diffusion_model

        res_dict = {1: [0, 1], 2: [0]}
        res_idx = 0
        for res in res_dict:
            for block in res_dict[res]:
                model.output_blocks[3*res +
                                    block][0].out_layers_features = injection[f'features{res_idx}'][step].to(device)
                res_idx += 1

        attn_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0]}
        attn_idx = 4
        for attn in attn_dict:
            for block in attn_dict[attn]:
                module = model.output_blocks[3*attn +
                                             block][1].transformer_blocks[0].attn1
                module.inject_q = injection[f'q{attn_idx}'][step].to(device)
                module.inject_k = injection[f'k{attn_idx}'][step].to(device)
                attn_idx += 1
