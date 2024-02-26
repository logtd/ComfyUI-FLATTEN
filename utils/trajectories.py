import random
from einops import rearrange
import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F


# TODO hard coded 512
def preprocess(img1_batch, img2_batch, transforms):
    img1_batch = F.resize(img1_batch, size=[512, 512], antialias=False)
    img2_batch = F.resize(img2_batch, size=[512, 512], antialias=False)
    return transforms(img1_batch, img2_batch)

#  "f h w c -> f c h w"


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


def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors


@torch.no_grad()
def sample_trajectories(frames, device):
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    clips = list(range(len(frames)))

    model = raft_large(weights=Raft_Large_Weights.DEFAULT,
                       progress=False).to(device)
    model = model.eval()
    frames = rearrange(frames,  "f h w c -> f c h w")
    current_frames, next_frames = preprocess(
        frames[clips[:-1]], frames[clips[1:]], transforms)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]

    predicted_flows = predicted_flows/512

    # TODO make resolution configurable
    # All these are needed with 512x512 due to upsample/downsample
    resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {64: 2,
                    32: 1,
                    16: 1,
                    8: 1}

    for resolution in resolutions:
        trajectories = {}
        predicted_flow_resolu = torch.round(resolution*torch.nn.functional.interpolate(
            predicted_flows, scale_factor=(resolution/512, resolution/512)))

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
                    # stupid padding with (-1, -1, -1)
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
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i) > 1]
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
        for p in list(left_points):
            useful_traj.append([p])

        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (
            window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index(
                        (t, x, y), window_sizes[resolution], H, W)
                    sequence = [(t, x, y)]+neighbours + [(0, 0, 0)
                                                         for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(
                        sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t, x, y)].copy()
                    traj.remove((t, x, y))
                    sequence = sequence + traj + \
                        [(0, 0, 0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1) **
                                  2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res
