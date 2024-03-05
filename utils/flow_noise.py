import torch


def create_noise_generator(directions_list, num_frames):
    def generator(latent_image):
        batch_size, c, h, w = latent_image.shape

        def create_noise(sigma, sigma_next):
            nonlocal latent_image
            visited = torch.zeros([num_frames, h, w], dtype=torch.bool)
            noise = torch.randn_like(latent_image[0])
            noise = torch.cat([noise.unsqueeze(0)]*num_frames)
            for t in range(num_frames):
                for y in range(h):
                    for x in range(w):
                        if visited[t, y, x]:
                            continue
                        for directions in directions_list:
                            if (t, x, y) in directions:
                                for (pt, px, py) in directions[(t, x, y)]:
                                    noise[pt, :, py, px] = noise[t, :, y, x]
                                    visited[pt, py, px] = True
                                break
            return noise
        return create_noise
    return generator
