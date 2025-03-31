# InfiniteYou-FLUX

Generate images that preserve your identity while applying any style or context you describe.

[![Replicate](https://replicate.com/infiniteyou-flux/badge)](https://replicate.com/infiniteyou-flux)

## What is InfiniteYou?

InfiniteYou is a model that takes a portrait image of you and creates a new image that:
- Keeps your facial identity
- Applies the style, setting, and context from your text prompt
- Maintains high image quality

Unlike other models, it doesn't just copy-paste your face - it rebuilds the entire image while preserving your likeness.

![Example of InfiniteYou](https://github.com/YOURNAME/infiniteyou-flux/blob/main/assets/examples/yangmi.jpg?raw=true)

## How to Use

### Python API Example

```python
import replicate

# Basic example
output = replicate.run(
    "infiniteyou-flux",
    input={
        "id_image": "https://example.com/your-photo.jpg",
        "prompt": "A portrait of a person in a sci-fi setting, cyberpunk style, neon lights",
        "model_version": "aes_stage2"
    }
)

# Advanced example with more parameters
output = replicate.run(
    "infiniteyou-flux",
    input={
        "id_image": "https://example.com/your-photo.jpg",
        "prompt": "An elegant portrait in Renaissance style, ornate period clothing, dramatic lighting",
        "model_version": "aes_stage2",
        "width": 960,
        "height": 1280,
        "num_steps": 40,
        "enable_realism": True,
        "seed": 42
    }
)

# Example with control image
output = replicate.run(
    "infiniteyou-flux",
    input={
        "id_image": "https://example.com/your-photo.jpg",
        "control_image": "https://example.com/pose-reference.jpg",
        "prompt": "A portrait, professional photography, high quality",
        "model_version": "sim_stage1"
    }
)

print(output)
```

### curl Example

```bash
# Remember to set your API token
export REPLICATE_API_TOKEN=your_token_here

# Basic example
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "2dc4b4516683d3639c3648d3c34294913fc11dc28396cd60fd86df877a04381b",
    "input": {
      "id_image": "https://example.com/your-photo.jpg",
      "prompt": "Portrait, 4K, high quality, cinematic",
      "model_version": "aes_stage2"
    }
  }' \
  https://api.replicate.com/v1/predictions
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| id_image | Upload a portrait image containing a human face | Required |
| prompt | Describe how you want the generated image to look | "Portrait, 4K, high quality, cinematic" |
| control_image | Optional second image to control the pose/position. Omit this parameter entirely when not needed. | None |
| model_version | Choose "aes_stage2" for better aesthetics or "sim_stage1" for higher identity similarity | "aes_stage2" |
| width | Output image width in pixels (recommended: 768, 864, or 960) | 864 |
| height | Output image height in pixels (recommended: 960, 1152, or 1280) | 1152 |
| num_steps | Number of diffusion steps (higher = better quality but slower) | 30 |
| guidance_scale | How closely to follow the prompt (higher = more prompt adherence) | 3.5 |
| seed | Random seed for reproducible results (None for random generation) | None |
| enable_realism | Apply the realism enhancement LoRA for more realistic results | false |
| enable_anti_blur | Apply the anti-blur LoRA to reduce blurriness | false |
| output_format | Choose the format of the output image (png, jpg, webp) | "webp" |
| output_quality | Set the quality for jpg and webp (1-100) | 80 |
| infusenet_conditioning_scale | Controls how strongly the identity image affects generation | 1.0 |
| infusenet_guidance_start | When to start applying identity guidance (0.0-0.1 recommended) | 0.0 |
| infusenet_guidance_end | When to stop applying identity guidance (usually keep at 1.0) | 1.0 |

## Tips for Best Results

1. **Clear Face Input**: Use a portrait where the face is clearly visible
2. **Detailed Prompts**: Be specific about style, setting, lighting, etc.
3. **Choose the Right Model**:
   - Use `aes_stage2` for better text-image alignment and aesthetics
   - Use `sim_stage1` for stronger identity preservation
4. **Experiment with Parameters**:
   - Try adjusting `infusenet_guidance_start` to 0.1 for higher identity similarity
   - For even more control, try slightly lowering `infusenet_conditioning_scale` to 0.9
   - The `enable_realism` LoRA can improve photorealism

## Examples

### Professional Portrait
```
Input: Your headshot
Prompt: "A professional portrait photograph, wearing a business suit, neutral background, studio lighting, 4K, sharp"
```

### Fantasy Character
```
Input: Your selfie
Prompt: "A fantasy character portrait, mythical setting, magical forest background, glowing aura, detailed fantasy clothing"
```

### Example Files
This repository includes example images in the `assets/examples` directory that you can use to test the model:
- `yangmi.jpg` - Female portrait example
- `yann-lecun_resize.jpg` - Male portrait example
- `man_pose.jpg` - Example pose control image

## Disclaimer and Licenses

Some images in this demo are from public domains or generated by models. These pictures are intended solely to show the capabilities of our research. If you have any concerns, please contact us, and we will promptly remove any inappropriate content.

The code in this demo is licensed under the [Apache License 2.0](./LICENSE), and our model is released under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](https://creativecommons.org/licenses/by-nc/4.0/legalcode) for academic research purposes only. Any manual or automatic downloading of the face models from [InsightFace](https://github.com/deepinsight/insightface), the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) base model, LoRAs ([Realism](https://civitai.com/models/631986?modelVersionId=706528) and [Anti-blur](https://civitai.com/models/675581/anti-blur-flux-lora)), *etc.*, must follow their original licenses and be used only for academic research purposes.

This research aims to positively impact the field of Generative AI. Users are granted the freedom to create images using this tool, but they must comply with local laws and use it responsibly. The developers do not assume any responsibility for potential misuse by users.

## Acknowledgements

InfiniteYou-FLUX was created by ByteDance Research. The original paper and more information can be found here:
- [InfiniteYou Project Page](https://bytedance.github.io/InfiniteYou)
- [Research Paper](https://arxiv.org/abs/2503.16418)
- [Original GitHub Repository](https://github.com/bytedance/InfiniteYou)
