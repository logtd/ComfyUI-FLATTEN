import torch


def _get_xl_resnets(model):
    obs = model.output_blocks
    return [obs[0][0], obs[1][0], obs[4][0], obs[5][0]]


def _get_xl_attns(model):
    obs = model.output_blocks
    transformers = [
        obs[2][1],
        obs[3][1],
        obs[4][1],
    ]
    attentions = []
    for transformer in transformers:
        for block in transformer.transformer_blocks:
            attentions.append(block.attn1)

    return attentions


def _is_xl(model):
    name = model.model._get_name()
    return name == 'SDXL' or name == 'PatchSDXL'


def _get_injection_names():
    names = ['features0', 'features1', 'features2']
    for i in range(4, 10):
        names.append(f'q{i}')
        names.append(f'k{i}')

    return names


def _clear_injections_xl(model):
    model = model.model.diffusion_model
    resnets = _get_xl_resnets(model)
    for resnet in resnets:
        resnet.out_layers_features = None

    attns = _get_xl_attns(model)
    for attn in attns:
        attn.q = None
        attn.k = None
        attn.inject_q = None
        attn.inject_k = None


# model.model._get_name() == 'SDXL'
def clear_injections(model):
    if _is_xl(model):
        _clear_injections_xl(model)
        return

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


def _get_blank_injection_dict_xl(context_windows, model):
    model = model.model.diffusion_model
    resnets = _get_xl_resnets(model)
    attns = _get_xl_attns(model)
    injection_dict = {}

    for i, resnet in enumerate(resnets):
        blank = {}
        for context_window in context_windows:
            blank[context_window[0]] = []
        injection_dict[f'res{i}'] = blank

    for i, attn in enumerate(attns):
        blank = {}
        for context_window in context_windows:
            blank[context_window[0]] = []
        injection_dict[f'q{i}'] = blank
        blank = {}
        for context_window in context_windows:
            blank[context_window[0]] = []
        injection_dict[f'k{i}'] = blank

    return injection_dict


def get_blank_injection_dict(context_windows, model):
    if _is_xl(model):
        return _get_blank_injection_dict_xl(context_windows, model)
    names = _get_injection_names()

    injection_dict = {}

    for name in names:
        blank = {}
        for context_window in context_windows:
            blank[context_window[0]] = []
        injection_dict[name] = blank
    return injection_dict


def _update_injections_xl(model, injection, context_start, save_steps):
    model = model.model.diffusion_model
    resnets = _get_xl_resnets(model)
    for i, res in enumerate(resnets):
        feature = res.out_layers_features.cpu()
        if len(injection[f'res{i}'][context_start]) < save_steps:
            injection[f'res{i}'][context_start].append(feature)

    attns = _get_xl_attns(model)
    for i, attn in enumerate(attns):
        if len(injection[f'q{i}'][context_start]) < save_steps:
            injection[f'q{i}'][context_start].append(attn.q.cpu())
            injection[f'k{i}'][context_start].append(attn.k.cpu())


def update_injections(model, injection, context_start, save_steps):
    if _is_xl(model):
        _update_injections_xl(model, injection, context_start, save_steps)
        return

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


def _inject_features_xl(model, injection, device, step, context_start, len_conds):
    model = model.model.diffusion_model

    resnets = _get_xl_resnets(model)
    for i, res in enumerate(resnets):
        feature = torch.cat(
            [injection[f'res{i}'][context_start][step][0, :, :].unsqueeze(0)]*len_conds)
        res.out_layers_features = feature.to(device)

    attns = _get_xl_attns(model)
    for i, attn in enumerate(attns):
        q = torch.cat(
            [injection[f'q{i}'][context_start][step]] * len_conds)
        attn.inject_q = q.to(device)
        k = torch.cat(
            [injection[f'k{i}'][context_start][step]] * len_conds)
        attn.inject_k = k.to(device)


def inject_features(model, injection, device, step, context_start, len_conds):
    if _is_xl(model):
        _inject_features_xl(model, injection, device,
                            step, context_start, len_conds)
        return

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
