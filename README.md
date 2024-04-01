# ComfyUI-FLATTEN
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
<img width="843" alt="flatten_nodes_screenshot" src="https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/3ea92d0c-b484-4290-9ed2-a07b5031a4c5">

* Node: Load Checkpoint with FLATTEN model
  * Loads any given SD1.5 checkpoint with the FLATTEN optical flow model. Use the `sdxl` branch of this repo to load SDXL models
  * The loaded model only works with the Flatten KSampler and a standard ComfyUI checkpoint loader is required for other KSamplers
 
* Node: Sample Trajectories
  * Takes the input images and samples their optical flow into trajectories. Trajectories are created for the dimensions of the input image and must match the latent size Flatten processes.
  * Context Length and Overlap for Batching with AnimateDiff-Evolved
    * Context Length defines the window size Flatten processes at a time. Flatten is not limitted to a certain frame count, but this can be used to reduce VRAM usage at a single time
    * Context Overlap is the overlap between windows
    * Can only use Standard Static from AnimateDiff-Evolved and these values must match the values given to AnimateDiff's Evolved Sampling context
    * Currently does not support Views
   
* Node: Unsampler (Flatten)
  * Unsamples the input latent and creates the needed injections required for sampling
  * Only use Euler or ddpm2m as the sampling method since this process creates noise from the input images
 
* Node: KSampler (Flatten)
  * Samples the unsampled latents and uses the injections from the Unsampler
  * Can use any sampling method, but use Euler or ddpm2m for editing pieces of the video or another sampling method to get drastic changes in the video
 
* Node: Apply Flatten Attention (SD1.5 Only)
  * Use Flatten's Optical Flow attention mechanism without the rest of Flatten's model -- can be used to combine with other models
  * Warning: Flatten's attention requires "Flow Noise" so it does not always work with methods that add normal noise
 
* Node: Create Flow Noise
  * Creates flow noise given a latent and trajectories
  * Can be used to add initial noise to a latent instead of using normal noise from a traditional KSampler


## Accompanying Node Repos
* [Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for loading and combining videos
* [AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) for batching options

## Examples
For working ComfyUI example workflows see the `example_workflows/` directory.

### Video Editing
FLATTEN excels at editing videos with temporal consistency. The recommended settings for this are to use an Unsampler and KSampler with `old_qk = 0`. The Unsampler should use the euler sampler and the KSampler should use the dpmpp_2m sampler. Users may experiment with `old_qk` depending on their use case, but it is not recommended to use other samplers or `add_noise` for video editing. Style transfer nodes such as IP-Adapter may have difficulty making quality edits without the additional noise and will require fine tuning.

### Scene Editing (Experimental)
Inspired by the optical flow use in FLATTEN, these nodes can utilize noise that is driven by optical flow. The current implementation is experimental and allows the user to create highly altered scenes, however it can lose some of the consistency and does not work well with high motion scenes.

To use this, it is recommended to use LCM on the KSampler (not the Unsampler) alongside setting `old_qk = 1` on the KSampler. Ancestral sampling methods also work well. Users may experiment with toggling the `add_noise` setting on the KSampler when using a sampling method that injects noise (e.g. anything besides Euler and dpmpp2). Using IPAdapter can help guide these generations towards a specific look.

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/18b49cbb-9647-48c0-9f3d-b58440fc9c1a

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/13769f9a-05f0-4669-ba80-556a8169e3df

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/f6fcf5c4-df0e-4ca4-8411-388520442d6c

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/d9942a82-aadb-49a6-92f4-9bf95de390ed

## ComfyUI Support
The ComfyUI-FLATTEN implementation can support most ComfyUI nodes, including ControlNets, IP-Adapter, LCM, InstanceDiffusion/GLIGEN, and many more.

### Batching
Currently batching for large amount of frames results in a loss in consistency and a possible solution is under consideration.

The current batching mechanism utilizes the AnimateDiff-Evolved batching nodes and is required to batch. See the example workflow for a working example.

### SDXL Support
Experiments for supporting SDXL were made and resulted in generating somewhat consistent videos, but not up-to-par with the SD1.5 implementation. 
Feel free to check out the `sdxl` branch, but there will be no further development in this direction.

### Unsupported
Currently the known unsupported custom ComfyUI features are:
* Scheduled Prompting
* Context Views for advanced batching

## Acknowledgements
* [Cong, Yuren and Xu, Mengmeng and Simon, Christian and Chen, Shoufa and Ren, Jiawei and Xie, Yanping and Perez-Rua, Juan-Manuel and Rosenhahn, Bodo and Xiang, Tao and He, Sen](https://github.com/yrcong/flatten) for their research on FLATTEN, producing the original repo, and contributing to open source.
* [Kosinkadink](https://github.com/Kosinkadink) for creating Video Helper Suite and AnimateDiff-Evolved
* [Kijai](https://github.com/kijai) for making helpful nodes
* [@AIWarper](https://twitter.com/AIWarper) for testing and making amazing content
