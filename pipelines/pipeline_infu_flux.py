# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers.models import FluxControlNetModel
from facexlib.recognition import init_recognition_model
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image

from .pipeline_flux_infusenet import FluxInfuseNetPipeline
from .resampler import Resampler


def seed_everything(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# modified from https://github.com/instantX-research/InstantID/blob/main/pipeline_stable_diffusion_xl_instantid.py
def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def extract_arcface_bgr_embedding(in_image, landmark, arcface_model=None, in_settings=None):
    kps = landmark
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    face_emb = arcface_model(arc_face_image)[0] # [512], normalized
    return face_emb


def resize_and_pad_image(source_img, target_img_size):
    # Get original and target sizes
    source_img_size = source_img.size
    target_width, target_height = target_img_size
    
    # Determine the new size based on the shorter side of target_img
    if target_width <= target_height:
        new_width = target_width
        new_height = int(target_width * (source_img_size[1] / source_img_size[0]))
    else:
        new_height = target_height
        new_width = int(target_height * (source_img_size[0] / source_img_size[1]))
    
    # Resize the source image using LANCZOS interpolation for high quality
    resized_source_img = source_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Compute padding to center resized image
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    # Create a new image with white background
    padded_img = Image.new("RGB", target_img_size, (255, 255, 255))
    padded_img.paste(resized_source_img, (pad_left, pad_top))
    
    return padded_img


class InfUFluxPipeline:
    def __init__(
            self, 
            base_model_path, 
            infu_model_path, 
            insightface_root_path = './',
            image_proj_num_tokens=8,
            infu_flux_version='v1.0',
            model_version='aes_stage2',
        ):

        self.infu_flux_version = infu_flux_version
        self.model_version = model_version
        
        # Load pipeline
        try:
            infusenet_path = os.path.join(os.path.dirname(infu_model_path), 'aes_stage2', 'InfuseNetModel')
            self.infusenet_aes = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
            infusenet_path = os.path.join(os.path.dirname(infu_model_path), 'sim_stage1', 'InfuseNetModel')
            self.infusenet_sim = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
        except:
            print("No InfiniteYou model found. Downloading from HuggingFace `ByteDance/InfiniteYou` to `./models/InfiniteYou` ...")
            snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir='./models/InfiniteYou', local_dir_use_symlinks=False)
            infu_model_path = os.path.join('./models/InfiniteYou', f'infu_flux_{infu_flux_version}', 'aes_stage2')
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet_aes = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
            infu_model_path = os.path.join('./models/InfiniteYou', f'infu_flux_{infu_flux_version}', 'sim_stage1')
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet_sim = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
            insightface_root_path = './models/InfiniteYou/supports/insightface'
        self.infusenet_sim.cpu()
        torch.cuda.empty_cache()
        try:
            pipe = FluxInfuseNetPipeline.from_pretrained(
                base_model_path,
                controlnet=self.infusenet_aes,
                torch_dtype=torch.bfloat16,
            )
        except:
            try:
                pipe = FluxInfuseNetPipeline.from_single_file(
                    base_model_path,
                    controlnet=self.infusenet_aes,
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                print(e)
                print('\nIf you are using `black-forest-labs/FLUX.1-dev` and have not downloaded it into a local directory, '
                      'please accept the agreement and obtain access at https://huggingface.co/black-forest-labs/FLUX.1-dev. '
                      'Then, use `huggingface-cli login` and your access tokens at https://huggingface.co/settings/tokens to authenticate. '
                      'After that, run the code again. If you have downloaded it, please use `base_model_path` to specify the correct path.')
                print('\nIf you are using other models, please download them to a local directory and use `base_model_path` to specify the correct path.')
                exit()
        pipe.to('cuda', torch.bfloat16)
        # CPU offload controlnet and T5 in advance
        pipe.controlnet.cpu()
        pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()
        pipe.text_encoder.to('cuda')
        pipe.transformer.to('cuda')
        pipe.vae.to('cuda')
        # pipe.enable_model_cpu_offload()
        self.pipe = pipe

        # Load image proj model
        num_tokens = image_proj_num_tokens
        image_emb_dim = 512
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        image_proj_model_path = os.path.join(os.path.dirname(infu_model_path), 'aes_stage2', 'image_proj_model.bin')
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        del ipm_state_dict
        self.image_proj_model_aes = copy.deepcopy(image_proj_model)
        self.image_proj_model_aes.to('cuda', torch.bfloat16)
        self.image_proj_model_aes.eval()
        
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        image_proj_model_path = os.path.join(os.path.dirname(infu_model_path), 'sim_stage1', 'image_proj_model.bin')
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        del ipm_state_dict
        self.image_proj_model_sim = image_proj_model
        self.image_proj_model_sim.to('cpu', dtype=torch.bfloat16)
        self.image_proj_model_sim.eval()

        self.image_proj_model = self.image_proj_model_aes

        # Load face encoder
        self.app_640 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))

        self.app_320 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))

        self.app_160 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))

        self.arcface_model = init_recognition_model('arcface', device='cuda')

        # Load LoRAs in advance
        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        self.loras_state_dict = {}
        self.loras_state_dict['realism'] = self.pipe._fetch_state_dict(os.path.join(os.path.dirname(insightface_root_path), 'optional_loras', 'flux_realism_lora.safetensors'),
            weight_name=None, use_safetensors=True, local_files_only=None, cache_dir=None, force_download=False, proxies=None, token=None, revision=None, subfolder=None, user_agent=user_agent, allow_pickle=True)
        self.loras_state_dict['anti_blur'] = self.pipe._fetch_state_dict(os.path.join(os.path.dirname(insightface_root_path), 'optional_loras', 'flux_anti_blur_lora.safetensors'),
            weight_name=None, use_safetensors=True, local_files_only=None, cache_dir=None, force_download=False, proxies=None, token=None, revision=None, subfolder=None, user_agent=user_agent, allow_pickle=True)

    def load_loras_state_dict(self, loras):
        names, scales = [],[]
        for lora_name, lora_scale in loras:
            print(f"loading lora state dict of {lora_name}")
            self.pipe.load_lora_weights(self.loras_state_dict[lora_name], adapter_name=lora_name)
            names.append(lora_name)
            scales.append(lora_scale)

        if len(names) > 0:
            self.pipe.set_adapters(names, adapter_weights=scales)
    
    def load_loras(self, loras):
        names, scales = [],[]
        for lora_path, lora_name, lora_scale in loras:
            if lora_path != "":
                print(f"loading lora {lora_path}")
                self.pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                names.append(lora_name)
                scales.append(lora_scale)

        if len(names) > 0:
            self.pipe.set_adapters(names, adapter_weights=scales)

    def _detect_face(self, id_image_cv2):
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        face_info = self.app_160.get(id_image_cv2)
        return face_info

    def __call__(
        self,
        id_image: Image.Image,  # PIL.Image.Image (RGB)
        prompt: str,
        control_image: Optional[Image.Image] = None,  # PIL.Image.Image (RGB) or None
        width = 864,
        height = 1152,
        seed = 42,
        guidance_scale = 3.5,
        num_steps = 30,
        infusenet_conditioning_scale = 1.0,
        infusenet_guidance_start = 0.0,
        infusenet_guidance_end = 1.0,
    ):        
        # Extract ID embeddings
        print('Preparing ID embeddings')
        id_image_cv2 = cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR)
        face_info = self._detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        landmark = face_info['kps']
        id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, self.arcface_model)
        id_embed = id_embed.clone().unsqueeze(0).float()
        id_embed = id_embed.reshape([1, -1, 512])
        id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        self.image_proj_model.to('cuda', torch.bfloat16)
        with torch.no_grad():
            id_embed = self.image_proj_model(id_embed)
            bs_embed, seq_len, _ = id_embed.shape
            id_embed = id_embed.repeat(1, 1, 1)
            id_embed = id_embed.view(bs_embed * 1, seq_len, -1)
            id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        
        # Load control image
        print('Preparing the control image')
        if control_image is not None:
            control_image = control_image.convert("RGB")
            control_image = resize_and_pad_image(control_image, (width, height))
            face_info = self._detect_face(cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR))
            if len(face_info) == 0:
                raise ValueError('No face detected in the control image')
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            control_image = draw_kps(control_image, face_info['kps'])
        else:
            out_img = np.zeros([height, width, 3])
            control_image = Image.fromarray(out_img.astype(np.uint8))

        # Perform inference
        print('Generating image')
        seed_everything(seed)
        image = self.pipe(
            prompt=prompt,
            controlnet_prompt_embeds=id_embed,
            control_image=control_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            controlnet_guidance_scale=1.0,
            controlnet_conditioning_scale=infusenet_conditioning_scale,
            control_guidance_start=infusenet_guidance_start,
            control_guidance_end=infusenet_guidance_end,
            height=height,
            width=width,
        ).images[0]

        return image
