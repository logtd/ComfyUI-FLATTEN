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
                 "inversion_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "injection_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "old_qk": ("INT", {"default": 0, "min": 0, "max": 1}),
                 "trajectories": ("TRAJECTORY",),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "inversion_positive": ("CONDITIONING", ),
                 "inversion_negative": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                 "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                 "return_with_leftover_noise": (["disable", "enable"], ),
                 }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, inversion_steps, injection_steps, old_qk, trajectories, cfg, sampler_name, scheduler, positive, negative, inversion_positive, inversion_negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        # PREPARTION
        latent = latent_image
        latent_image = latent["samples"]
        latent_image = rearrange(latent_image, "(b f) c h w -> b c f h w", b=1)
        transformer_options = {
            'flatten': {
                'trajs': trajectories,
                'old_qk': old_qk
            }
        }
        model['transformer_options'] = transformer_options
        # UNSAMPLER
        device = comfy.model_management.get_torch_device()

        end_at_step = inversion_steps

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

        inversion_positive = comfy.sample.convert_cond(inversion_positive)
        inversion_negative = comfy.sample.convert_cond(inversion_negative)

        models, inference_memory = comfy.sample.get_additional_models(
            inversion_positive, inversion_negative, model.model_dtype())

        comfy.model_management.load_models_gpu(
            [model] + models, model.memory_required(noise.shape) + inference_memory)

        sampler = comfy.samplers.KSampler(real_model, steps=inversion_steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sigmas = sampler.sigmas.flip(0) + 0.0001

        pbar = comfy.utils.ProgressBar(inversion_steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, inversion_steps + steps)

        # DO A CLEANUP HERE JUST IN CASE
        for i in range(len(sigmas[:-1])):
            inversion_sigmas = [sigmas[i], sigmas[i+1]]
            noise = sampler.sample(noise, inversion_positive, inversion_negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False,
                                   denoise_mask=noise_mask, sigmas=inversion_sigmas, start_step=0, last_step=end_at_step, callback=callback)
            # GRAB INJECTIONS HERE

        noise = noise.cpu()

        comfy.sample.cleanup_additional_models(models)

        latent_image = noise

        # REVERSE INJECTIONS HERE!!!!

        # SAMPLER ######

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

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(inversion_steps + step +
                                 1, inversion_steps + steps)

        for i in range(len(sigmas)):
            if i < injection_steps:
                continue  # DO INJECTION HERE
            samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False,
                                     denoise_mask=noise_mask, sigmas=[sigmas[i], sigmas[i+1]], start_step=0, last_step=end_at_step, callback=callback)
        samples = samples.cpu()

        comfy.sample.cleanup_additional_models(models)

        out = latent.copy()
        out["samples"] = samples

        return (out, )
