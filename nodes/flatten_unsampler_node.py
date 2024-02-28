import comfy.samplers
import torch


class UnsamplerFlattenNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
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

    def unsampler(self, model, sampler_name, steps, end_at_step, scheduler, noramlize, positive, latent_image, trajectories, old_qk):
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
        original_transformer_options = model.model_options.get(
            'transformer_options', {})

        transformer_options = {
            **original_transformer_options,
            'flatten': {
                'trajs': trajectories,
                'old_qk': old_qk,
                'original_shape': original_shape
            }
        }
        model.model_options['transformer_options'] = transformer_options

        # SETUP NOISE
        noise = torch.zeros(latent_image.size(
        ), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        # SETUP SAMPLING
        sampler = comfy.samplers.KSampler(model.model, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        ksampler = comfy.samplers.ksampler(sampler_name)

        sigmas = sigmas = sampler.sigmas.flip(0) + 0.0001
        injection_dict = self._get_blank_injection_dict(steps)
        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)
            # model.model_options['transformer_options']['ad_params']['sub_idxs']
            transformer_options = model.model_options.get(
                'transformer_options', {})
            ad_params = transformer_options.get('ad_params', {})
            sub_idxs = ad_params.get('sub_idxs', None)
            if sub_idxs is None:
                sub_idxs = list(range(original_shape[0]))
            self._update_injections(model, injection_dict, step, sub_idxs)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # UNSAMPLE MODEL
        self._clear_injections(model)
        samples = comfy.sample.sample_custom(model, noise, cfg, ksampler, sigmas, positive, negative,
                                             latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        # CLEANUP
        self._clear_injections(model)
        model.model_options['transformer_options'] = original_transformer_options
        del transformer_options
        del callback

        # RETURN SAMPLES
        if normalize:
            # technically doesn't normalize because unsampling is not guaranteed to end at a std given by the schedule
            samples -= samples.mean()
            samples /= samples.std()

        out = latent.copy()
        out['samples'] = samples
        return (out, injection_dict)

    def _get_blank_injection_dict(self, steps):
        return {
            'features0': [None] * steps,
            'features1': [None] * steps,
            'features2': [None] * steps,
            'q4': [None] * steps,
            'k4': [None] * steps,
            'q5': [None] * steps,
            'k5': [None] * steps,
            'q6': [None] * steps,
            'k6': [None] * steps,
            'q7': [None] * steps,
            'k7': [None] * steps,
            'q8': [None] * steps,
            'k8': [None] * steps,
            'q9': [None] * steps,
            'k9': [None] * steps,
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

    def _update_injections(self, model, injection, step, idxs):
        model = model.model.diffusion_model

        res_dict = {1: [0, 1], 2: [0]}
        res_idx = 0
        for res in res_dict:
            for block in res_dict[res]:
                feature = model.output_blocks[3*res +
                                              block][0].out_layers_features.cpu()
                if injection[f'features{res_idx}'][step] is not None:
                    injection[f'features{res_idx}'][step] = torch.cat(
                        [injection[f'features{res_idx}'][step], feature], dim=2)
                else:
                    injection[f'features{res_idx}'][step] = feature
                res_idx += 1

        attn_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0]}
        attn_idx = 4
        for attn in attn_dict:
            for block in attn_dict[attn]:
                module = model.output_blocks[3*attn +
                                             block][1].transformer_blocks[0].attn1
                q = module.q.cpu()
                if injection[f'q{attn_idx}'][step] is not None:
                    injection[f'q{attn_idx}'][step] = torch.cat(
                        injection[f'q{attn_idx}'][step], q)
                else:
                    injection[f'q{attn_idx}'][step] = q
                k = module.k.cpu()
                if injection[f'k{attn_idx}'][step] is not None:
                    injection[f'k{attn_idx}'][step] = torch.cat(
                        injection[f'k{attn_idx}'][step], k)
                else:
                    injection[f'k{attn_idx}'][step] = k

                attn_idx += 1
