from .nodes.load_flatten_model_node import FlattenCheckpointLoaderNode
from .nodes.flatten_ksampler_node import KSamplerFlattenNode
from .nodes.trajectory_node import TrajectoryNode


NODE_CLASS_MAPPINGS = {
    "FlattenCheckpointLoaderNode": FlattenCheckpointLoaderNode,
    "KSamplerFlattenNode": KSamplerFlattenNode,
    "TrajectoryNode": TrajectoryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlattenCheckpointLoaderNode": "Load Checkpoint with FLATTEN model",
    "KSamplerFlattenNode": "KSampler (Flatten)",
    "TrajectoryNode": "Sample Trajectories",
}
