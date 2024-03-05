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
                for x in range(h):
                    for y in range(w):
                        if visited[t, x, y]:
                            continue
                        for directions in directions_list:
                            if (t, x, y) in directions:
                                for (pt, px, py) in directions[(t, x, y)]:
                                    noise[pt, :, px, py] = noise[t, :, x, y]
                                    visited[pt, px, py] = True
                                break
            return noise
        return create_noise
    return generator
