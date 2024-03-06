import comfy.samplers
import torch
import comfy.k_diffusion.sampling

from ..utils.injection_utils import get_blank_injection_dict, clear_injections, update_injections
from ..utils.flow_noise import create_noise_generator


class UnsamplerFlattenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "save_steps": ("INT", {"default": 8, "min": 0, "max": 10000}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "normalize": (["disable", "enable"], ),
                 "positive": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "trajectories": ("TRAJECTORY", ),
                 "old_qk": ("INT", {"default": 0, "min": 0, "max": 1}),
                 }}

    RETURN_TYPES = ("LATENT", "INJECTIONS")
    FUNCTION = "unsampler"

    CATEGORY = "sampling"

    def unsampler(self, model, sampler_name, steps, save_steps, scheduler, normalize, positive, latent_image, trajectories, old_qk):
        # DEFAULTS
        device = comfy.model_management.get_torch_device()

        cfg = 1  # hardcoded to make attention injection faster and simpler
        noise_seed = 777  # no noise is added
        negative = []
        normalize = normalize == 'enable'

        latent = latent_image
        latent_image = latent["samples"]
        original_shape = latent_image.shape

        # SETUP TRANSFORMER OPTIONS
        injection_dict = get_blank_injection_dict(
            trajectories['context_windows'])

        def save_injections_handler(context_start):
            update_injections(model, injection_dict, context_start, save_steps)

        original_transformer_options = model.model_options.get(
            'transformer_options', {})

        transformer_options = {
            **original_transformer_options,
            'flatten': {
                'trajs_windows': trajectories['trajectory_windows'],
                'old_qk': old_qk,
                'input_shape': original_shape,
                'stage': 'inversion',
                'save_injections_handler': save_injections_handler
            }
        }
        model.model_options['transformer_options'] = transformer_options

        # SETUP NOISE
        default_noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler
        comfy.k_diffusion.sampling.default_noise_sampler = create_noise_generator(
            [traj['directions'] for traj in trajectories['trajectory_windows'].values()], latent_image.shape[0])
        noise = torch.zeros(latent_image.size(
        ), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(
                latent["noise_mask"], latent_image.shape, device)

        # SETUP SAMPLING
        sampler = comfy.samplers.KSampler(model.model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        ksampler = comfy.samplers.ksampler(sampler_name)

        sigmas = sigmas = sampler.sigmas.flip(0) + 0.0001

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # UNSAMPLE MODEL
        try:
            clear_injections(model)
            samples = comfy.sample.sample_custom(model, noise, cfg, ksampler, sigmas, positive, negative,
                                                 latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
        except Exception as e:
            print('Flatten Unsampler error encountereed:', e)
            raise e
        finally:
            # CLEANUP
            clear_injections(model)
            comfy.k_diffusion.sampling.default_noise_sampler = default_noise_sampler
            model.model_options['transformer_options'] = original_transformer_options
            del transformer_options
            del callback
            del save_injections_handler

        # RETURN SAMPLES
        if normalize:
            # technically doesn't normalize because unsampling is not guaranteed to end at a std given by the schedule
            samples -= samples.mean()
            samples /= samples.std()

        out = latent.copy()
        out['samples'] = samples
        return (out, injection_dict)
