from .nodes.load_flatten_model_node import FlattenCheckpointLoaderNode
from .nodes.flatten_ksampler_node import KSamplerFlattenNode
from .nodes.flatten_unsampler_node import UnsamplerFlattenNode
from .nodes.trajectory_node import TrajectoryNode


NODE_CLASS_MAPPINGS = {
    "FlattenCheckpointLoaderNode": FlattenCheckpointLoaderNode,
    "KSamplerFlattenNode": KSamplerFlattenNode,
    "UnsamplerFlattenNode": UnsamplerFlattenNode,
    "TrajectoryNode": TrajectoryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlattenCheckpointLoaderNode": "Load Checkpoint with FLATTEN model",
    "KSamplerFlattenNode": "KSampler (Flatten)",
    "UnsamplerFlattenNode": "Unsampler (Flatten)",
    "TrajectoryNode": "Sample Trajectories",
}
