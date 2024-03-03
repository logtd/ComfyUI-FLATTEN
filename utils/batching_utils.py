# Adjusted from ADE: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved

def create_windows_static_standard(num_frames, context_length, overlap):
    windows = []
    if num_frames <= context_length or context_length == 0:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = context_length - overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + context_length
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(
                list(range(final_start_idx, final_start_idx + context_length)))
            break
        windows.append(list(range(start_idx, start_idx + context_length)))
    return windows
