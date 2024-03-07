# ComfyUI-FLATTEN (Work in Progress)
ComfyUI nodes to use FLATTEN.

Original research repo: [FLATTEN](https://github.com/yrcong/flatten)

## Table of Contents
- [Installation](#installation)
  - [How to Install](#how-to-install)
- [Nodes](#nodes)
- [Accompanying Node Repos](#accompanying-node-repos)
- [Examples](#examples)
- [Acknowledgements](#acknowledgements)

## Installation

### How to Install
TODO
Clone or download this repo into your `ComfyUI/custom_nodes/` directory.
Then run `pip install -r requirments.txt` within the cloned repo.

## Nodes
TODO

## Accompanying Node Repos
* [Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## Video Editing
TODO

## Scene Editing (Experimental)
Not included in the original FLATTEN was the use of generating completely new scenes through noise injection that is driven by optical flow. The current implementation is experimental and allows the user to create completely new scenes with details, however it can lose some of the consistency and does not work well with high motion scenes.

To use this, it is recommended to use LCM on the KSampler (not the Unsampler) alongside setting `old_qk = 1` on the KSampler. Ancestral sampling methods also work well.

https://github.com/logtd/ComfyUI-FLATTEN/assets/160989552/18b49cbb-9647-48c0-9f3d-b58440fc9c1a





## Acknowledgements
* [Cong, Yuren and Xu, Mengmeng and Simon, Christian and Chen, Shoufa and Ren, Jiawei and Xie, Yanping and Perez-Rua, Juan-Manuel and Rosenhahn, Bodo and Xiang, Tao and He, Sen](https://github.com/yrcong/flatten) for their research on FLATTEN, producing the original repo, and contributing to open source.
* [Kosinkadink](https://github.com/Kosinkadink) for creating Video Helper Suite
* [Kijai](https://github.com/kijai) for testing and his node suites
