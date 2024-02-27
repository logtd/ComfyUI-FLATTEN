from einops import rearrange


def transform_to_2d(x):
    b = x.shape[0]
    return rearrange(x, 'b c f h w -> (b f) c h w'), b


def transformed_to_3d(x, b):
    return rearrange(x, '(b f) c h w -> b c f h w', b=b)


def apply_unet_patch3d(h, hs, transformer_options, patch):
    h, bh = transform_to_2d(h)
    hs, bhs = transform_to_2d(hs)
    h, hs = patch(h, hs, transformer_options)
    h = transformed_to_3d(h, bh)
    hs = transformed_to_3d(hs, bhs)
    return h, hs


def apply_patch3d(x, transformer_options, patch):
    x, b = transform_to_2d(x)
    x = patch(x, transformer_options)
    x = transformed_to_3d(x, b)
    return x
