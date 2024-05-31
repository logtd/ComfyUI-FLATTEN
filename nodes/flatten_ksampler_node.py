import torch
import comfy.sd
import comfy.model_base
import comfy.samplers
import comfy.k_diffusion.sampling

from ..utils.injection_utils import inject_features, clear_injections
from ..utils.flow_noise import create_noise_generator


class KSamplerFlattenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "add_noise": (["disable", "enable"], ),
                 "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                 "injection_steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                 "old_qk": ("INT", {"default": 0, "min": 0, "max": 1}),
                 "trajectories": ("TRAJECTORY",),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "positive": ("CONDITIONING", ),
                 "negative": ("CONDITIONING", ),
                 "latent_image": ("LATENT", ),
                 "injections": ("INJECTIONS",),
                 "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                 "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                 "return_with_leftover_noise": (["disable", "enable"], ),
                 }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    injection_step = 0
    previous_timestep = None

    def sample(self, model, add_noise, noise_seed, steps, injection_steps, old_qk, trajectories, cfg, sampler_name, scheduler, positive, negative, latent_image, injections, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        # DEFAULTS
        device = comfy.model_management.get_torch_device()

        latent = latent_image
        latent_image = latent["samples"]
        original_shape = latent_image.shape

        # SETUP NOISE
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(
                latent["noise_mask"], latent_image.shape, device)

        add_noise = add_noise == 'enable'
        if not add_noise:
            noise = torch.zeros(latent_image.size(
            ), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(
                latent_image, noise_seed, batch_inds)
            noise = torch.cat([noise[0].unsqueeze(0)] * original_shape[0])

        # SETUP SIGMAS AND STEPS
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sampler.sigmas
        timestep_to_step = {}
        for i, sigma in enumerate(sigmas):
            t = int(model.model.model_sampling.timestep(sigma))
            timestep_to_step[t] = i

        # FLATTEN TRANSFORMER OPTIONS
        original_transformer_options = model.model_options.get(
            'transformer_options', {})

        # step hack
        self.previous_timestep = None
        self.injection_step = -1

        def injection_handler(sigma, context_start, len_conds):
            clear_injections(model)
            t = int(model.model.model_sampling.timestep(sigma))
            if self.previous_timestep != t:
                self.previous_timestep = t
                self.injection_step += 1
            if self.injection_step < injection_steps:
                inject_features(model, injections, device,
                                self.injection_step, context_start, len_conds)
            else:
                clear_injections(model)
            return self.injection_step

        transformer_options = {
            **original_transformer_options,
            'flatten': {
                'trajs_windows': trajectories['trajectory_windows'],
                'old_qk': old_qk,
                'injection_handler': injection_handler,
                'input_shape': original_shape,
                'stage': 'sampling',
                'injection_steps': injection_steps
            }
        }
        model.model_options['transformer_options'] = transformer_options

        # HACK NOISE
        default_noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler
        comfy.k_diffusion.sampling.default_noise_sampler = create_noise_generator(
            [traj['directions'] for traj in trajectories['trajectory_windows'].values()], latent_image.shape[0])

        # SAMPLE MODEL
        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        try:
            clear_injections(model)
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                          denoise=denoise, disable_noise=False, start_step=start_at_step, last_step=end_at_step,
                                          force_full_denoise=not return_with_leftover_noise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
        except Exception as e:
            print('Flatten KSampler error encountereed:', e)
            raise e
        finally:
            # CLEANUP
            clear_injections(model)
            comfy.k_diffusion.sampling.default_noise_sampler = default_noise_sampler
            model.model_options['transformer_options'] = original_transformer_options
            self.previous_timestep = None
            self.injection_step = 0

            del injection_handler
            del transformer_options

        # RETURN
        out = {}
        out["samples"] = samples
        return (out, )
