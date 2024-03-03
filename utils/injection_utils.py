import torch


def _get_injection_names():
    names = ['features0', 'features1', 'features2']
    for i in range(4, 10):
        names.append(f'q{i}')
        names.append(f'k{i}')

    return names


def clear_injections(model):
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


def get_blank_injection_dict(context_windows):
    names = _get_injection_names()

    injection_dict = {}

    for name in names:
        blank = {}
        for context_window in context_windows:
            blank[context_window[0]] = []
        injection_dict[name] = blank
    return injection_dict


def update_injections(model, injection, context_start, save_steps):
    model = model.model.diffusion_model

    res_dict = {1: [0, 1], 2: [0]}
    res_idx = 0
    for res in res_dict:
        for block in res_dict[res]:
            feature = model.output_blocks[3*res +
                                          block][0].out_layers_features.cpu()
            if len(injection[f'features{res_idx}'][context_start]) < save_steps:
                injection[f'features{res_idx}'][context_start].append(feature)
            res_idx += 1

    attn_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0]}
    attn_idx = 4
    for attn in attn_dict:
        for block in attn_dict[attn]:
            module = model.output_blocks[3*attn +
                                         block][1].transformer_blocks[0].attn1
            if len(injection[f'q{attn_idx}'][context_start]) < save_steps:
                injection[f'q{attn_idx}'][context_start].append(module.q.cpu())
                injection[f'k{attn_idx}'][context_start].append(module.k.cpu())

            attn_idx += 1


def inject_features(model, injection, device, step, context_start, len_conds):
    model = model.model.diffusion_model

    res_dict = {1: [0, 1], 2: [0]}
    res_idx = 0
    for res in res_dict:
        for block in res_dict[res]:
            feature = torch.cat(
                [injection[f'features{res_idx}'][context_start][step][0, :, :].unsqueeze(0)]*len_conds)
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
                [injection[f'q{attn_idx}'][context_start][step]] * len_conds)
            module.inject_q = q.to(device)
            k = torch.cat(
                [injection[f'k{attn_idx}'][context_start][step]] * len_conds)
            module.inject_k = k.to(device)
            attn_idx += 1
