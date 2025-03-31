# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
from typing import Optional
import torch
import gc
import pillow_avif
import subprocess
from cog import BasePredictor, Input, Path as CogPath
from huggingface_hub import snapshot_download
from pillow_heif import register_heif_opener
from PIL import Image

from pipelines.pipeline_infu_flux import InfUFluxPipeline

# Register HEIF support for Pillow
register_heif_opener()

class ModelVersion:
    STAGE_1 = "sim_stage1"
    STAGE_2 = "aes_stage2"
    DEFAULT_VERSION = STAGE_2



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # subprocess.check_output(["pget", "https://storage.googleapis.com/replicate-hf-weights/infiniteyou/models.tar", "models.tar"], close_fds=True)
        # subprocess.check_output(["tar", "-xvf", "models.tar"])

        # Initialize the pipeline with default configuration
        self.model_version = ModelVersion.DEFAULT_VERSION
        self.enable_realism = False
        self.enable_anti_blur = False
        
        model_path = f'./models/InfiniteYou/infu_flux_v1.0/{self.model_version}'
        print(f'Loading model from {model_path}')
        
        self.pipeline = InfUFluxPipeline(
            base_model_path='./models/FLUX.1-dev',
            infu_model_path=model_path,
            insightface_root_path='./models/InfiniteYou/supports/insightface',
            image_proj_num_tokens=8,
            infu_flux_version='v1.0',
            model_version=self.model_version,
        )

    def _prepare_pipeline(self, model_version, enable_realism, enable_anti_blur):
        # Update pipeline if model version has changed
        if model_version != self.model_version:
            print(f'Switching model to {model_version}')
            self.model_version = model_version
            if model_version == 'aes_stage2':
                self.pipeline.infusenet_sim.cpu()
                self.pipeline.image_proj_model_sim.cpu()
                torch.cuda.empty_cache()
                self.pipeline.infusenet_aes.to('cuda')
                self.pipeline.pipe.controlnet = self.pipeline.infusenet_aes
                self.pipeline.image_proj_model_aes.to('cuda')
                self.pipeline.image_proj_model = self.pipeline.image_proj_model_aes
            else:
                self.pipeline.infusenet_aes.cpu()
                self.pipeline.image_proj_model_aes.cpu()
                torch.cuda.empty_cache()
                self.pipeline.infusenet_sim.to('cuda')
                self.pipeline.pipe.controlnet = self.pipeline.infusenet_sim
                self.pipeline.image_proj_model_sim.to('cuda')
                self.pipeline.image_proj_model = self.pipeline.image_proj_model_sim

        # Update LoRA settings
        self.enable_realism = enable_realism
        self.enable_anti_blur = enable_anti_blur
        
        self.pipeline.pipe.delete_adapters(['realism', 'anti_blur'])
        loras = []
        if enable_realism: 
            loras.append(['realism', 1.0])
        if enable_anti_blur: 
            loras.append(['anti_blur', 1.0])
        
        self.pipeline.load_loras_state_dict(loras)

    def predict(
        self,
        # Required inputs
        id_image: CogPath = Input(
            description="Upload a portrait image containing a human face. For multiple faces, only the largest face will be detected."
        ),
        
        # Common inputs
        prompt: str = Input(
            description="Describe how you want the generated image to look. Be specific about details, style, background, etc.", 
            default="Portrait, 4K, high quality, cinematic"
        ),
        control_image: Optional[CogPath] = Input(
            description="Optional: Upload a second image to control the pose/position of the face in the output", 
            default=None
        ),
        model_version: str = Input(
            description="Choose the model version - 'aes_stage2' for better text-image alignment and aesthetics, 'sim_stage1' for higher identity similarity", 
            choices=["sim_stage1", "aes_stage2"],
            default="aes_stage2"
        ),
        
        # Image quality and dimensions
        width: int = Input(
            description="Output image width in pixels (recommended: 768, 864, or 960)", 
            default=864, 
            ge=256, 
            le=1280
        ),
        height: int = Input(
            description="Output image height in pixels (recommended: 960, 1152, or 1280)", 
            default=1152, 
            ge=256, 
            le=1280
        ),
        
        # Generation parameters
        num_steps: int = Input(
            description="Number of diffusion steps - higher values (30-50) give better quality but take longer", 
            default=30, 
            ge=1, 
            le=100
        ),
        guidance_scale: float = Input(
            description="How closely to follow the prompt (higher = more prompt adherence, lower = more freedom)", 
            default=3.5, 
            ge=0.0, 
            le=10.0
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results (None generates a random seed)", 
            default=None
        ),
        
        # Optional enhancements
        enable_realism: bool = Input(
            description="Apply the realism enhancement LoRA for more realistic-looking results", 
            default=False
        ),
        enable_anti_blur: bool = Input(
            description="Apply the anti-blur LoRA to reduce blurriness in the results", 
            default=False
        ),
        
        output_format: str = Input(
            description="Choose the format of the output image", choices=["png", "jpg", "webp"], default="webp"
        ),
        output_quality: int = Input(
            description="Set the quality of the output image for jpg and webp (1-100)", ge=1, le=100, default=80
        ),
        
        # Advanced parameters (typically don't need adjustment)
        infusenet_conditioning_scale: float = Input(
            description="Advanced: Controls how strongly the identity image affects generation (lower values = less identity preservation)", 
            default=1.0, 
            ge=0.0, 
            le=1.0
        ),
        infusenet_guidance_start: float = Input(
            description="Advanced: When to start applying identity guidance (0.0-0.1 recommended)", 
            default=0.0, 
            ge=0.0, 
            le=1.0
        ),
        infusenet_guidance_end: float = Input(
            description="Advanced: When to stop applying identity guidance (usually keep at 1.0)", 
            default=1.0, 
            ge=0.0, 
            le=1.0
        ),
    ) -> CogPath:
        """Generate a portrait image that maintains the identity/likeness of the input image while applying the style described in your prompt"""
        
        # Ensure the pipeline is using the correct model and settings
        self._prepare_pipeline(model_version, enable_realism, enable_anti_blur)
        
        # Use random seed if None is provided
        if seed is None:
            seed = torch.seed() & 0xFFFFFFFF
        print(f"Using seed: {seed}") # Log the seed being used
            
        # Open the image files before passing them to the pipeline
        id_img = Image.open(id_image)
        control_img = Image.open(control_image) if control_image is not None else None
        
        # Generate the image
        output_image = self.pipeline(
            id_image=id_img,
            prompt=prompt,
            control_image=control_img,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            infusenet_conditioning_scale=infusenet_conditioning_scale,
            infusenet_guidance_start=infusenet_guidance_start,
            infusenet_guidance_end=infusenet_guidance_end,
        )
        
        # Before saving, ensure image is in RGB mode
        if output_image.mode != 'RGB':
            output_image = output_image.convert('RGB')
        
        # Prepare saving arguments
        extension = output_format.lower()
        if extension == "jpg":
            extension = "jpeg" # PIL uses 'jpeg'
            
        output_path_str = f"/tmp/output.{extension}"
        save_kwargs = {}
        if extension in ["jpeg", "webp"]:
            save_kwargs["quality"] = output_quality
            # optimize is useful for reducing file size, especially for jpeg
            save_kwargs["optimize"] = True 
            print(f"Saving as {extension.upper()} with quality {output_quality}")
        else:
             print(f"Saving as {extension.upper()}")


        # Save the output image
        output_image.save(output_path_str, **save_kwargs)
        
        # Return a CogPath object created from the saved file
        return CogPath(output_path_str)
