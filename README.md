# ComfyUI-FLATTEN (Work in Progress)
ComfyUI nodes to use FLATTEN.

Original research repo: [FLATTEN](https://github.com/yrcong/flatten)

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/518865fe-8bf3-44aa-ab05-edaaff92c3e0

## Table of Contents
- [Installation](#installation)
  - [How to Install](#how-to-install)
- [Nodes](#nodes)
- [Accompanying Node Repos](#accompanying-node-repos)
- [Examples](#examples)
- [Acknowledgements](#acknowledgements)

## Installation

### How to Install
Clone or download this repo into your `ComfyUI/custom_nodes/` directory or use the ComfyUI-Manager to automatically install the nodes. No additional Python packages outside of ComfyUI requirements should be necessary.

## Nodes
TODO

## Accompanying Node Repos
* [Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## Examples
For working ComfyUI example workflows see the `example_workflows/` directory.

### Video Editing
FLATTEN excels at editing videos with temporal consistency. The recommended settings for this are to use an Unsampler and KSampler with `old_qk = 0`. The Unsampler should use the euler sampler and the KSampler should use the dpmpp_2m sampler. Users may experiment with `old_qk` depending on their use case, but it is not recommended to use other samplers or `add_noise` for video editing. Style transfer nodes such as IP-Adapter may have difficulty making quality edits without the additional noise and will require fine tuning.

### Scene Editing (Experimental)
Inspired by the optical flow use in FLATTEN, these nodes can utilize noise that is driven by optical flow. The current implementation is experimental and allows the user to create highly altered scenes, however it can lose some of the consistency and does not work well with high motion scenes.

To use this, it is recommended to use LCM on the KSampler (not the Unsampler) alongside setting `old_qk = 1` on the KSampler. Ancestral sampling methods also work well. Users may experiment with toggling the `add_noise` setting on the KSampler when using a sampling method that injects noise (e.g. anything besides Euler and dpmpp2). Using IPAdapter can help guide these generations towards a specific look.

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/18b49cbb-9647-48c0-9f3d-b58440fc9c1a

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/13769f9a-05f0-4669-ba80-556a8169e3df

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/d191c01b-1085-477f-a1c1-4610b204d32c

## ComfyUI Support
The ComfyUI-FLATTEN implementation can support most ComfyUI nodes, including ControlNets, IP-Adapter, LCM, InstanceDiffusion/GLIGEN, and many more.

### Batching
Currently batching for large amount of frames results in a loss in consistency and a possible solution is under consideration.

The current batching mechanism utilizes the AnimateDiff-Evolved batching nodes and is required to batch. 

### SDXL Support
Experiments for supporting SDXL were made and resulted in generating somewhat consistent videos, but not up-to-par with the SD1.5 implementation. 
Feel free to check out the `sdxl` branch, but there will be no further development in this direction.

### Unsupported
Currently the known unsupported custom ComfyUI features are:
* Scheduled Prompting

## Acknowledgements
* [Cong, Yuren and Xu, Mengmeng and Simon, Christian and Chen, Shoufa and Ren, Jiawei and Xie, Yanping and Perez-Rua, Juan-Manuel and Rosenhahn, Bodo and Xiang, Tao and He, Sen](https://github.com/yrcong/flatten) for their research on FLATTEN, producing the original repo, and contributing to open source.
* [Kosinkadink](https://github.com/Kosinkadink) for creating Video Helper Suite
<<<<<<< HEAD
* [@AIWarper](https://x.com/AIWarper) for testing and sharing amazing content
* [Kijai](https://github.com/kijai) for creating useful ComfyUI nodes
  
=======
* [Kijai](https://github.com/kijai) for making helpful nodes
* [@AIWarper](https://twitter.com/AIWarper) for testing and making amazing content
>>>>>>> ab554f2 (Update README.md)
