import random
from einops import rearrange
import torch

import torchvision.transforms.functional as F


# TODO hard coded 512
def preprocess(img1_batch, img2_batch, transforms):
    img1_batch = F.resize(img1_batch, size=[512, 512], antialias=False)
    img2_batch = F.resize(img2_batch, size=[512, 512], antialias=False)
    return transforms(img1_batch, img2_batch)


def keys_with_same_value(dictionary):
    result = {}
    for key, value in dictionary.items():
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

    conflict_points = {}
    for k in result.keys():
        if len(result[k]) > 1:
            conflict_points[k] = result[k]
    return conflict_points


def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


def neighbors_index(point, h_size, w_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-h_size, h_size + 1):
        for j in range(-w_size, w_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors


def get_window_size(resolution):
    # this isn't always correct and needs an actual calculation
    if resolution > 64:
        return 4
    elif resolution > 32:
        return 2
    else:
        return 1


@torch.no_grad()
def sample_trajectories(frames, model, weights, device):
    model.eval()
    transforms = weights.transforms()
    image_height = frames.shape[1]
    image_width = frames.shape[2]

    clips = list(range(len(frames)))
    frames = rearrange(frames,  "f h w c -> f c h w")
    # current_frames, next_frames = preprocess(
    #     frames[clips[:-1]], frames[clips[1:]], transforms)
    current_frames, next_frames = frames[clips[:-1]], frames[clips[1:]]
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]

    predicted_flows[:, 0] = predicted_flows[:, 0]/image_height
    predicted_flows[:, 1] = predicted_flows[:, 1]/image_width

    height_reso = image_height//8
    height_resoultions = [height_reso]
    width_reso = image_width//8
    width_resolutions = [width_reso]

    res = {}

    for height_resolution, width_resolution in zip(height_resoultions, width_resolutions):
        trajectories = {}
        x_flows = torch.round(height_resolution*torch.nn.functional.interpolate(
            predicted_flows[:, 1].unsqueeze(1), scale_factor=(height_resolution/image_height, width_resolution/image_width)))
        y_flows = torch.round(width_resolution*torch.nn.functional.interpolate(
            predicted_flows[:, 0].unsqueeze(1), scale_factor=(height_resolution/image_height, width_resolution/image_width)))

        predicted_flow_resolu = torch.cat([y_flows, x_flows], dim=1)

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)] = (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point] = (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x % H): 0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj += [[pixel]
                            for pixel in pixel_set if pixel_set[pixel] == 0]
        # these are vectors from point start to point end [(t,x,y), (t+1, x,y)...]
        all_traj += active_traj

        useful_traj = [segment for segment in all_traj if len(segment) > 1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(
            trajs)) == 0, "There should not be duplicates in the useful trajectories."

        all_points = set([(t, x, y) for t in range(T)
                         for x in range(H) for y in range(W)])
        left_points = all_points - set(trajs)
        for p in list(left_points):  # add points that are missing
            useful_traj.append([p])

        longest_length = max([len(traj) for traj in useful_traj])
        h_size = get_window_size(height_resolution)
        w_size = get_window_size(width_resolution)
        window_size = (h_size*2+1) * (w_size*2+1)
        sequence_length = window_size + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}  # point to vector/segmeent
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index(
                        (t, x, y), h_size, w_size, H, W)
                    sequence = [(t, x, y)]+neighbours + [(0, 0, 0)
                                                         for i in range(window_size-1-len(neighbours))]
                    sequence_mask = torch.zeros(
                        sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t, x, y)].copy()
                    traj.remove((t, x, y))
                    sequence = sequence + traj + \
                        [(0, 0, 0) for k in range(longest_length-1-len(traj))
                         ]  # add (0,0,0) to fill in gaps
                    sequence_mask[window_size:window_size + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        seqs = torch.cat([seqs[:, 0, :].unsqueeze(
            1), seqs[:, -len(frames)+1:, :]], dim=1)
        seqs = rearrange(seqs, '(f n) l d -> f n l d', f=len(frames))
        masks = torch.stack(masks)
        masks = torch.cat([masks[:, 0].unsqueeze(
            1), masks[:, -len(frames)+1:]], dim=1)
        masks = rearrange(masks, '(f n) l -> f n l', f=len(frames))
        res["traj{}".format(height_resolution)] = seqs.cpu()
        res["mask{}".format(height_resolution)] = masks.cpu()
    return res
