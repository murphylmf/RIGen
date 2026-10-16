<div align="center">

<h1>RIGen: Beyond Action Following<br>with Reciprocal Interaction in Generative Gameplay</h1>

<p><strong>Semantic closed-loop video generation for interactive FPS worlds</strong></p>

<p>
  <strong>Mengfei Li¹,²</strong> ·
  <strong>Yi Zhang²<sup>†</sup></strong> ·
  <strong>Xingtong Ge¹,²</strong> ·
  <strong>Lunjie Zhu¹,²</strong> ·
  <strong>Fangyu Lin¹,²</strong> ·
  <strong>Peng Li¹</strong> ·
  <strong>Guanglu Song²</strong> ·
  <strong>Yu Liu²</strong> ·
  <strong>Wei Xue¹<sup>‡</sup></strong> ·
  <strong>Wenhan Luo¹</strong> ·
  <strong>Yike Guo¹</strong>
</p>

<p>¹ The Hong Kong University of Science and Technology · ² Vivix Group Limited</p>

<p><sub><sup>†</sup>Project Lead · <sup>‡</sup>Corresponding Author</sub></p>

</div>

[![Project Page](https://img.shields.io/badge/Project-Page-orange.svg?logo=googlechrome&logoColor=white)](https://murphylmf.github.io/RIGen/)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Murphyyyy/RIGen-Xonotic-Dataset)
[![RIGen Model](https://img.shields.io/badge/Hugging%20Face-RIGen-blue.svg?logo=huggingface&logoColor=white)](https://huggingface.co/Murphyyyy/RIGen)
[![Data Engine](https://img.shields.io/badge/Data%20Engine-GitHub-181717.svg?logo=github&logoColor=white)](https://github.com/murphylmf/RIGen-Xonotic-Data-Engine)

## Installation

Use Python 3.11 with a CUDA-enabled PyTorch installation:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The runtime needs CUDA and FlashAttention 4. Install a FlashAttention 4 build compatible with your PyTorch and CUDA versions, then check it with:

~~~bash
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4: OK')"
~~~

## Download weights

Run these commands from the repository root:

~~~bash
mkdir -p .model_downloads weights

hf download Murphyyyy/RIGen weights/model.safetensors \
  --local-dir .model_downloads/RIGen
cp .model_downloads/RIGen/weights/model.safetensors weights/model.safetensors

hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-dev.safetensors \
  ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --local-dir .model_downloads/LTX-2.3
cp .model_downloads/LTX-2.3/ltx-2.3-22b-dev.safetensors weights/
cp .model_downloads/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors weights/
cp .model_downloads/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors weights/

hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir .model_downloads/gemma-3-12b-it-qat-q4_0-unquantized
mkdir -p weights/gemma-3-12b-it-qat-q4_0-unquantized
cp -a .model_downloads/gemma-3-12b-it-qat-q4_0-unquantized/. \
  weights/gemma-3-12b-it-qat-q4_0-unquantized/
~~~

The Gemma repository is gated on Hugging Face. Request access and run hf auth login before downloading it.

The [RIGen-Xonotic-Data-Engine](https://github.com/murphylmf/RIGen-Xonotic-Data-Engine) is the separate pipeline for creating training data.

## Run inference

The default player controller is VLM-driven. For each 16-frame rollout block, it reads the latest generated view and chooses movement, view direction, and primary fire.

### 1. Start the VLM service

Open a second terminal, activate the same environment, and run:

~~~bash
vllm serve OpenGVLab/InternVL3-8B \
  --served-model-name internvl3-8b \
  --trust-remote-code \
  --port 8012
~~~

The command below expects the VLM service at http://127.0.0.1:8012/v1 with model name internvl3-8b. If you use a different address or model name, pass the matching values with --vlm-base-url and --vlm-model.

### 2. Run the verified test case

From the repository root, run:

~~~bash
python inference/run_rigen.py \
  --case-id test_case \
  --initial-enemy-hp 50 \
  --rollout-blocks 6 \
  --output-dir outputs/test_case
~~~

The case files are in `cases/test_case/`. The rollout, final video, generated `actions.json`, and per-run prompt context are written below `outputs/test_case/`.

## Run with custom actions

To provide the player actions yourself, use --no-player-controller and point to an action file:

~~~bash
python inference/run_rigen.py \
  --case-id test_case \
  --initial-enemy-hp 50 \
  --rollout-blocks 6 \
  --no-player-controller \
  --no-visual-feedback \
  --action-file actions/example.json \
  --output-dir outputs/test_case_custom
~~~

This mode does not need the VLM service. The input file is `actions/example.json`; the run writes the resulting `actions.json` to the output directory. Each block has matching axes and buttons lists with 16 rows.

Each axis row is [left_x, left_y, right_x, right_y]:

| Position | Meaning | Negative / positive |
| --- | --- | --- |
| axes[0] | left stick horizontal movement | strafe left / strafe right |
| axes[1] | left stick vertical movement | move forward / move backward |
| axes[2] | right stick horizontal view | look left / look right |
| axes[3] | right stick vertical view | look up / look down |

Each button row is [fire, aim, melee, jump, reload, Y]:

| Position | Meaning |
| --- | --- |
| buttons[0] | primary fire |
| buttons[1] | unused; keep 0 |
| buttons[2] | unused; keep 0 |
| buttons[3] | jump |
| buttons[4] | reload |
| buttons[5] | unused Y button; keep 0 |

Axis values are in [-1, 1]; button values are normally 0 or 1. If the rollout is longer than the supplied list, blocks repeat.

## Citation

~~~bibtex
@misc{li2026rigenactionfollowingreciprocal,
      title={RIGen: Beyond Action Following with Reciprocal Interaction in Generative Gameplay},
      author={Mengfei Li and Yi Zhang and Xingtong Ge and Lunjie Zhu and Fangyu Lin and Peng Li and Guanglu Song and Yu Liu and Wei Xue and Wenhan Luo and Yike Guo},
      year={2026},
      eprint={2610.20811},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.20811},
}
~~~

<div align="center">
  <br>
  <a href="https://murphylmf.github.io/RIGen/">RIGen Project Page</a>
  ·
  <a href="https://huggingface.co/Murphyyyy/RIGen">RIGen Model</a>
</div>
