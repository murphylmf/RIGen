Place files downloaded from the two Hugging Face model repositories here:

- `model.safetensors` for RIGen from `Murphyyyy/RIGen`.
Download the LTX-2.3 foundation files from `Lightricks/LTX-2.3` and the Gemma text encoder from `google/gemma-3-12b-it-qat-q4_0-unquantized`, as described in the root README. The expected paths are:

- `ltx-2.3-22b-dev.safetensors`
- `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- `gemma-3-12b-it-qat-q4_0-unquantized/`

The inference scripts resolve every filename relative to this project root.
