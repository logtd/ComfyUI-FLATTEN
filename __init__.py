from .nodes.load_flatten_model_node import FlattenCheckpointLoaderNode
from .nodes.flatten_ksampler_node import KSamplerFlattenNode
from .nodes.flatten_unsampler_node import UnsamplerFlattenNode
from .nodes.trajectory_node import TrajectoryNode
from .nodes.apply_flatten_attention_node import ApplyFlattenAttentionNode
from .nodes.create_flow_noise_node import CreateFlowNoiseNode


NODE_CLASS_MAPPINGS = {
    "FlattenCheckpointLoaderNode": FlattenCheckpointLoaderNode,
    "KSamplerFlattenNode": KSamplerFlattenNode,
    "UnsamplerFlattenNode": UnsamplerFlattenNode,
    "TrajectoryNode": TrajectoryNode,
    "ApplyFlattenAttentionNode": ApplyFlattenAttentionNode,
    "CreateFlowNoiseNode": CreateFlowNoiseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlattenCheckpointLoaderNode": "Load Checkpoint with FLATTEN model",
    "KSamplerFlattenNode": "KSampler (Flatten)",
    "UnsamplerFlattenNode": "Unsampler (Flatten)",
    "TrajectoryNode": "Sample Trajectories",
    "ApplyFlattenAttentionNode": "Apply Flatten Attention",
    "CreateFlowNoiseNode": "Create Flow Noise"
}
