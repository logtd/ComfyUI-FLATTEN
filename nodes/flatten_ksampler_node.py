import torch
import comfy.sd
import comfy.model_base
import comfy.samplers


class KSamplerFlattenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "add_noise": (["disable", "enable"], ),
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
                latent["noise_mask"], noise.shape, device)

        add_noise = add_noise == 'enable'
        if not add_noise:
            noise = torch.zeros(latent_image.size(
            ), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(
                latent_image, noise_seed, batch_inds)

        # SETUP SIGMAS AND STEPS
        sampler = comfy.samplers.KSampler(model.model, steps=steps, device=device, sampler=sampler_name,
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
        self.injection_step = 0

        def injection_handler(sigma, idxs, len_conds):
            if idxs is None:
                idxs = list(range(original_shape[0]))
            self._clear_injections(model)
            t = int(model.model.model_sampling.timestep(sigma))
            if self.previous_timestep != t:
                self.previous_timestep = t
                self.injection_step += 1
            if self.injection_step < injection_steps:
                self._inject(model, injections, device,
                             self.injection_step, idxs, len_conds)
            else:
                self._clear_injections(model)

        transformer_options = {
            **original_transformer_options,
            'flatten': {
                'trajs': trajectories,
                'old_qk': old_qk,
                'injection_handler': injection_handler,
                'original_shape': original_shape
            }
        }
        model.model_options['transformer_options'] = transformer_options

        # SAMPLE MODEL
        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        self._clear_injections(model)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                      denoise=denoise, disable_noise=False, start_step=start_at_step, last_step=end_at_step,
                                      force_full_denoise=not return_with_leftover_noise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        # CLEANUP
        self._clear_injections(model)
        model.model_options['transformer_options'] = original_transformer_options
        self.previous_timestep = None
        self.injection_step = 0

        del injection_handler
        del transformer_options

        # RETURN
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

    def _inject(self, model, injection, device, step, idxs, len_conds):
        model = model.model.diffusion_model

        res_dict = {1: [0, 1], 2: [0]}
        res_idx = 0
        for res in res_dict:
            for block in res_dict[res]:
                feature = torch.cat(
                    [injection[f'features{res_idx}'][step][0, :, idxs].unsqueeze(0)]*len_conds)
                model.output_blocks[3*res +
                                    block][0].out_layers_features = feature.to(device)
                res_idx += 1

        attn_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0]}
        attn_idx = 4
        for attn in attn_dict:
            for block in attn_dict[attn]:
                module = model.output_blocks[3*attn +
                                             block][1].transformer_blocks[0].attn1
                q = torch.cat(
                    [injection[f'q{attn_idx}'][step][idxs]] * len_conds)
                module.inject_q = q.to(device)
                k = torch.cat(
                    [injection[f'k{attn_idx}'][step][idxs]] * len_conds)
                module.inject_k = k.to(device)
                attn_idx += 1
