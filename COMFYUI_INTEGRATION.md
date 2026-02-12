# ComfyUI Integration Guide for OmniVideo2

This guide describes how to integrate the OmniVideo2 video editing model into ComfyUI as custom nodes, enabling users to perform video-to-video editing and text-to-video generation through ComfyUI's visual workflow interface.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Overview](#architecture-overview)
4. [Custom Node Implementation](#custom-node-implementation)
5. [Installation Instructions](#installation-instructions)
6. [Usage in ComfyUI](#usage-in-comfyui)
7. [Optimization and Best Practices](#optimization-and-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

ComfyUI is a node-based interface for Stable Diffusion and other generative AI models. Integrating OmniVideo2 into ComfyUI involves creating custom nodes that:

1. **Load the OmniVideo2 model** - Handle model initialization with all required components (DiT, VAE, T5 encoder, Qwen3-VL)
2. **Process inputs** - Accept text prompts, source videos, and generation parameters
3. **Generate outputs** - Produce edited or generated videos using the OmniVideo2 pipeline
4. **Manage GPU memory** - Handle model offloading for efficient memory usage

### Key Components of OmniVideo2

- **DiT Backbone (Dual Models)**: High-noise and low-noise diffusion transformers (14B or 1.3B parameters)
- **VAE**: Wan2.1 VAE for encoding/decoding video latents
- **T5 Text Encoder**: T5-XXL for text embedding
- **Qwen3-VL Vision-Language Model**: For visual feature extraction and caption expansion
- **Mixed Condition Mechanism**: Combines text, visual latents, and optional control signals

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **PyTorch**: 2.8.0 or higher with CUDA support
- **NVIDIA GPU**: Recommended 80GB VRAM for OmniVideo2-A14B, or 24GB+ for OmniVideo2-1.3B
- **ComfyUI**: Latest version installed and working

### Required Dependencies

All dependencies from OmniVideo2's `requirements.txt`:
```bash
torch==2.8.0
torchvision==0.23.0
transformers==4.57.6
diffusers==0.35.1
accelerate==1.10.1
einops==0.8.1
decord==0.6.0
imageio==2.37.0
pillow==11.0.0
flash_attn==2.8.3  # Optional but recommended
qwen-vl-utils
```

### Model Checkpoints

Download and organize model checkpoints:

```
${CKPT_DIR}/
├── high_noise_model/
│   └── model.pt              # High-noise timestep DiT model
├── low_noise_model/
│   └── model.pt              # Low-noise timestep DiT model
├── special_tokens.pkl        # Special token embeddings
├── models_t5_umt5-xxl-enc-bf16.pth  # T5 encoder weights
└── Wan2.1_VAE.pth           # VAE model
```

Additionally, download Qwen3-VL model:
- [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)

---

## Architecture Overview

### OmniVideo2 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing                          │
├─────────────────────────────────────────────────────────────┤
│ • Text Prompt (edit instruction)                             │
│ • Source Video (for video-to-video editing)                  │
│ • Generation Parameters (resolution, frames, steps, etc.)    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Vision-Language Processing                      │
├─────────────────────────────────────────────────────────────┤
│ Qwen3-VL Model:                                              │
│ • Reads source video + edit instruction                      │
│ • Generates detailed target caption                          │
│ • Extracts visual features for conditioning                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Text Encoding (T5)                            │
├─────────────────────────────────────────────────────────────┤
│ • Encodes expanded caption to text embeddings                │
│ • Handles special tokens for conditioning                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              VAE Encoding (Source Video)                     │
├─────────────────────────────────────────────────────────────┤
│ • Encode source video frames to latent space                 │
│ • Provides visual conditioning signal                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Diffusion Generation (DiT Models)                   │
├─────────────────────────────────────────────────────────────┤
│ • High-noise model: Initial denoising steps                  │
│ • Low-noise model: Final refinement steps                    │
│ • Mixed cross-attention with text + visual conditioning      │
│ • Flow matching with UniPC/DDIM/Euler solver                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  VAE Decoding                                │
├─────────────────────────────────────────────────────────────┤
│ • Decode latents to pixel space                              │
│ • Output video frames                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
                  OUTPUT VIDEO
```

### Key Classes and Methods

1. **`OmniVideoX2XUnified`** (`omnivideo/x2x_gen_unified.py`):
   - Main pipeline class
   - Method: `generate()` - Core generation function
   - Handles model initialization, loading, and inference

2. **`generate_caption_and_extract_features()`** (`omnivideo/vllm_model.py`):
   - Processes source video and prompt through Qwen3-VL
   - Returns expanded caption and visual features

3. **Model Components**:
   - `T5EncoderModel` - Text encoding
   - `Wan2_1_VAE` - VAE encoding/decoding
   - `UnifiedWanWithMixedConditionModel` - DiT backbone

---

## Custom Node Implementation

### Node Structure

Create a ComfyUI custom node package with the following structure:

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-OmniVideo2/
        ├── __init__.py              # Node registration
        ├── omnivideo2_nodes.py      # Main node implementations
        ├── utils.py                 # Helper functions
        ├── requirements.txt         # Dependencies
        └── README.md                # Installation instructions
```

### Node Classes to Implement

#### 1. **OmniVideo2ModelLoader** Node

Loads and initializes the OmniVideo2 model with all components.

```python
class OmniVideo2ModelLoader:
    """
    Loads OmniVideo2 model with all required components.
    
    Outputs a MODEL object that contains:
    - OmniVideoX2XUnified pipeline
    - Qwen3-VL model and processor
    - Configuration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_dir": ("STRING", {"default": "/path/to/checkpoints"}),
                "qwen3vl_path": ("STRING", {"default": "/path/to/Qwen3-VL"}),
                "model_variant": (["OmniVideo2-A14B", "OmniVideo2-1.3B"], {"default": "OmniVideo2-A14B"}),
                "device_id": ("INT", {"default": 0, "min": 0, "max": 7}),
            },
            "optional": {
                "use_cpu_offload": ("BOOLEAN", {"default": True}),
                "init_on_cpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("OMNIVIDEO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "OmniVideo2"
    
    def load_model(self, checkpoint_dir, qwen3vl_path, model_variant, 
                   device_id, use_cpu_offload=True, init_on_cpu=True):
        """
        Initialize OmniVideo2 pipeline.
        """
        import torch
        from omnivideo.configs import WAN_CONFIGS
        from omnivideo.x2x_gen_unified import OmniVideoX2XUnified
        from omnivideo.vllm_model import load_qwen3vl_model_and_processor
        
        # Select appropriate config
        if "1.3B" in model_variant:
            config_name = "v2v-1.3B"
        else:
            config_name = "v2v-A14B"
        
        config = WAN_CONFIGS[config_name]
        
        # Initialize OmniVideo model
        omni_model = OmniVideoX2XUnified(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=0,
            init_on_cpu=init_on_cpu,
        )
        
        # Load Qwen3-VL for visual understanding
        qwen_model, qwen_processor = load_qwen3vl_model_and_processor(
            qwen3vl_path,
            device_id=device_id
        )
        
        model_dict = {
            "omni_model": omni_model,
            "qwen_model": qwen_model,
            "qwen_processor": qwen_processor,
            "config": config,
            "device_id": device_id,
            "use_cpu_offload": use_cpu_offload,
        }
        
        return (model_dict,)
```

#### 2. **OmniVideo2VideoToVideo** Node

Performs video-to-video editing with text prompts.

```python
class OmniVideo2VideoToVideo:
    """
    Video-to-video editing using OmniVideo2.
    
    Takes a source video and edit instruction, outputs edited video.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OMNIVIDEO_MODEL",),
                "source_video": ("VIDEO",),  # ComfyUI video type
                "edit_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "frame_num": ("INT", {"default": 41, "min": 5, "max": 121, "step": 4}),
                "sample_fps": ("INT", {"default": 8, "min": 1, "max": 60}),
                "sampling_steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "guide_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "sample_solver": (["unipc", "ddim", "euler"], {"default": "unipc"}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("edited_video",)
    FUNCTION = "generate_video"
    CATEGORY = "OmniVideo2"
    
    def generate_video(self, model, source_video, edit_prompt, width, height,
                      frame_num, sample_fps, sampling_steps, guide_scale, 
                      shift, seed, negative_prompt="", sample_solver="unipc"):
        """
        Generate edited video using OmniVideo2 pipeline.
        """
        import torch
        import numpy as np
        from omnivideo.vllm_model import (
            generate_caption_and_extract_features,
            offload_qwen3vl_to_cpu,
            load_qwen3vl_to_gpu
        )
        
        omni_model = model["omni_model"]
        qwen_model = model["qwen_model"]
        qwen_processor = model["qwen_processor"]
        device_id = model["device_id"]
        use_cpu_offload = model["use_cpu_offload"]
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 0xffffffffffffffff, (1,)).item()
        
        # Step 1: Offload OmniVideo to CPU if needed
        if use_cpu_offload:
            self._offload_omnivideo_to_cpu(omni_model)
            torch.cuda.empty_cache()
        
        # Step 2: Load Qwen3-VL and generate caption + features
        load_qwen3vl_to_gpu(qwen_model, device_id)
        
        caption, visual_emb = generate_caption_and_extract_features(
            qwen_model=qwen_model,
            processor=qwen_processor,
            source_video_path=source_video,  # Need to convert from ComfyUI format
            edit_prompt=edit_prompt,
            device_id=device_id
        )
        
        # Step 3: Offload Qwen3-VL and load OmniVideo back
        offload_qwen3vl_to_cpu(qwen_model)
        torch.cuda.empty_cache()
        
        if use_cpu_offload:
            self._load_omnivideo_to_gpu(omni_model, device_id)
        
        # Step 4: Encode source video with VAE
        source_latents = self._encode_video_to_latents(
            omni_model, source_video, width, height, frame_num
        )
        
        # Step 5: Generate video using diffusion
        output_latents = omni_model.generate(
            input_prompt=caption,
            visual_emb=visual_emb,
            ar_vision_input=source_latents,
            size=(width, height),
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt,
            seed=seed,
            precision_dtype=torch.bfloat16,
            offload_model=use_cpu_offload,
        )
        
        # Step 6: Decode latents to video
        output_video = self._decode_latents_to_video(
            omni_model, output_latents, sample_fps
        )
        
        return (output_video,)
    
    def _offload_omnivideo_to_cpu(self, model):
        """Helper to move OmniVideo components to CPU"""
        if hasattr(model, 'high_noise_model') and model.high_noise_model:
            model.high_noise_model.to('cpu')
        if hasattr(model, 'low_noise_model') and model.low_noise_model:
            model.low_noise_model.to('cpu')
    
    def _load_omnivideo_to_gpu(self, model, device_id):
        """Helper to move OmniVideo components to GPU"""
        device = torch.device(f"cuda:{device_id}")
        if hasattr(model, 'high_noise_model') and model.high_noise_model:
            model.high_noise_model.to(device)
    
    def _encode_video_to_latents(self, model, video, width, height, frame_num):
        """Encode video frames to VAE latent space"""
        # Convert ComfyUI video format to tensor
        # Process through VAE encoder
        # Return latent tensor
        pass
    
    def _decode_latents_to_video(self, model, latents, fps):
        """Decode VAE latents to video frames"""
        # Decode latents through VAE
        # Convert to ComfyUI video format
        # Return video
        pass
```

#### 3. **OmniVideo2TextToVideo** Node

Generates video from text prompts only (without source video).

```python
class OmniVideo2TextToVideo:
    """
    Text-to-video generation using OmniVideo2.
    
    Takes a text prompt and generates video from scratch.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OMNIVIDEO_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "frame_num": ("INT", {"default": 41, "min": 5, "max": 121, "step": 4}),
                "sample_fps": ("INT", {"default": 8, "min": 1, "max": 60}),
                "sampling_steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "guide_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "sample_solver": (["unipc", "ddim", "euler"], {"default": "unipc"}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("generated_video",)
    FUNCTION = "generate_video"
    CATEGORY = "OmniVideo2"
    
    def generate_video(self, model, prompt, width, height, frame_num, 
                      sample_fps, sampling_steps, guide_scale, shift, seed,
                      negative_prompt="", sample_solver="unipc"):
        """
        Generate video from text using OmniVideo2 pipeline.
        Similar to VideoToVideo but without source video conditioning.
        """
        # Implementation similar to VideoToVideo but:
        # - No source video encoding
        # - ar_vision_input = None
        # - Pure text-to-video generation
        pass
```

#### 4. **OmniVideo2SaveVideo** Node

Saves generated video to disk.

```python
class OmniVideo2SaveVideo:
    """
    Save OmniVideo2 generated video to disk.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "output_path": ("STRING", {"default": "./output"}),
                "filename_prefix": ("STRING", {"default": "omnivideo2"}),
            },
            "optional": {
                "format": (["mp4", "avi", "mov"], {"default": "mp4"}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_video"
    CATEGORY = "OmniVideo2"
    
    def save_video(self, video, output_path, filename_prefix, format="mp4"):
        """
        Save video to disk with proper encoding.
        """
        import os
        import imageio
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{format}"
        filepath = os.path.join(output_path, filename)
        
        # Save video using imageio
        # video should be in format [frames, height, width, channels]
        imageio.mimwrite(filepath, video, fps=8, codec='libx264')
        
        return {"ui": {"video": [filepath]}}
```

### Node Registration (`__init__.py`)

```python
"""
ComfyUI-OmniVideo2 Custom Nodes
Video editing and generation using OmniVideo2
"""

from .omnivideo2_nodes import (
    OmniVideo2ModelLoader,
    OmniVideo2VideoToVideo,
    OmniVideo2TextToVideo,
    OmniVideo2SaveVideo,
)

NODE_CLASS_MAPPINGS = {
    "OmniVideo2ModelLoader": OmniVideo2ModelLoader,
    "OmniVideo2VideoToVideo": OmniVideo2VideoToVideo,
    "OmniVideo2TextToVideo": OmniVideo2TextToVideo,
    "OmniVideo2SaveVideo": OmniVideo2SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVideo2ModelLoader": "Load OmniVideo2 Model",
    "OmniVideo2VideoToVideo": "OmniVideo2 Video Edit",
    "OmniVideo2TextToVideo": "OmniVideo2 Text-to-Video",
    "OmniVideo2SaveVideo": "Save OmniVideo2 Video",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

---

## Installation Instructions

### Step 1: Install ComfyUI

If you haven't already, install ComfyUI:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### Step 2: Create Custom Node Directory

```bash
cd ComfyUI/custom_nodes
mkdir ComfyUI-OmniVideo2
cd ComfyUI-OmniVideo2
```

### Step 3: Add OmniVideo2 Code

Copy the OmniVideo2 repository into the custom node or install as dependency:

**Option A: Copy omnivideo package**
```bash
# From OmniVideo2 repo root
cp -r omnivideo /path/to/ComfyUI/custom_nodes/ComfyUI-OmniVideo2/
```

**Option B: Install as editable package**
```bash
# From OmniVideo2 repo root
pip install -e .
```

### Step 4: Install Dependencies

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-OmniVideo2
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # Optional but recommended
```

### Step 5: Download Model Checkpoints

1. Download OmniVideo2 checkpoints from HuggingFace:
   - [OmniVideo2-A14B](https://huggingface.co/Fudan-FUXI/OmniVideo2-A14B)
   - [OmniVideo2-1.3B](https://huggingface.co/Fudan-FUXI/OmniVideo2-1.3B)

2. Download Qwen3-VL:
   - [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)

3. Organize checkpoints as described in [Prerequisites](#model-checkpoints)

### Step 6: Restart ComfyUI

```bash
cd /path/to/ComfyUI
python main.py
```

Your OmniVideo2 nodes should now appear in the node menu under the "OmniVideo2" category.

---

## Usage in ComfyUI

### Basic Video-to-Video Editing Workflow

1. **Load Model**:
   - Add "Load OmniVideo2 Model" node
   - Set `checkpoint_dir` to your checkpoint path
   - Set `qwen3vl_path` to your Qwen3-VL model path
   - Choose model variant (A14B or 1.3B)

2. **Load Source Video**:
   - Add "Load Video" node (built-in ComfyUI node)
   - Select your source video file

3. **Edit Video**:
   - Add "OmniVideo2 Video Edit" node
   - Connect model output from step 1
   - Connect video from step 2
   - Enter edit prompt (e.g., "Change the dog to a cat")
   - Adjust parameters (resolution, frames, steps, guidance)

4. **Save Output**:
   - Add "Save OmniVideo2 Video" node
   - Connect edited video output
   - Set output path and filename

5. **Execute Workflow**:
   - Click "Queue Prompt" to run the workflow
   - Monitor progress in the console

### Basic Text-to-Video Generation Workflow

1. **Load Model** (same as above)

2. **Generate Video**:
   - Add "OmniVideo2 Text-to-Video" node
   - Connect model output
   - Enter text prompt describing desired video
   - Adjust generation parameters

3. **Save Output** (same as above)

### Example Workflow JSON

```json
{
  "1": {
    "class_type": "OmniVideo2ModelLoader",
    "inputs": {
      "checkpoint_dir": "/path/to/checkpoints",
      "qwen3vl_path": "/path/to/Qwen3-VL",
      "model_variant": "OmniVideo2-A14B",
      "device_id": 0,
      "use_cpu_offload": true
    }
  },
  "2": {
    "class_type": "LoadVideo",
    "inputs": {
      "video": "source_video.mp4"
    }
  },
  "3": {
    "class_type": "OmniVideo2VideoToVideo",
    "inputs": {
      "model": ["1", 0],
      "source_video": ["2", 0],
      "edit_prompt": "Change the dog to a cat",
      "width": 832,
      "height": 480,
      "frame_num": 41,
      "sample_fps": 8,
      "sampling_steps": 40,
      "guide_scale": 3.0,
      "shift": 5.0,
      "seed": 42
    }
  },
  "4": {
    "class_type": "OmniVideo2SaveVideo",
    "inputs": {
      "video": ["3", 0],
      "output_path": "./outputs",
      "filename_prefix": "edited_video"
    }
  }
}
```

---

## Optimization and Best Practices

### Memory Management

**Problem**: OmniVideo2 requires significant GPU memory, especially with the A14B model and Qwen3-VL.

**Solutions**:

1. **Enable CPU Offloading**:
   ```python
   # In ModelLoader node
   use_cpu_offload = True
   init_on_cpu = True
   ```
   - Offloads OmniVideo2 during Qwen3-VL processing
   - Offloads inactive DiT model (high/low noise) during generation
   - Reduces peak memory usage at cost of speed

2. **Use Smaller Model**:
   - OmniVideo2-1.3B requires ~24GB VRAM vs 80GB for A14B
   - Slight quality reduction but much more accessible

3. **Reduce Resolution**:
   ```python
   width = 640  # Instead of 832
   height = 368  # Instead of 480
   ```

4. **Reduce Frame Count**:
   ```python
   frame_num = 25  # Instead of 41 (must be 4n+1)
   ```

5. **Use Flash Attention**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```
   - Reduces memory usage during attention computation
   - Speeds up inference

### Performance Optimization

1. **Preload Models**:
   - Load model once at startup
   - Reuse across multiple generations
   - Avoid reloading between generations

2. **Batch Processing**:
   - Process multiple videos in sequence
   - Amortize model loading overhead

3. **Optimal Sampling Settings**:
   ```python
   sampling_steps = 30-40  # Good quality/speed tradeoff
   sample_solver = "unipc"  # Fastest high-quality solver
   guide_scale = 3.0        # Lower = faster, higher = more faithful
   ```

4. **Use Torch Compile** (PyTorch 2.0+):
   ```python
   # Add to model initialization
   model = torch.compile(model, mode="reduce-overhead")
   ```

### Quality Optimization

1. **Caption Expansion**:
   - Qwen3-VL expands sparse prompts into detailed captions
   - More detailed input prompts = better results

2. **Visual Conditioning Strength**:
   - For subtle edits: use stronger source conditioning
   - For major transformations: reduce source influence

3. **Guidance Scale**:
   - Lower (2.0-3.0): More creative, less faithful to prompt
   - Higher (4.0-7.0): More faithful to prompt, may be less natural

4. **Sampling Steps**:
   - Minimum 30 steps recommended
   - 40-50 steps for best quality
   - Beyond 50 shows diminishing returns

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms**: CUDA out of memory error during generation

**Solutions**:
- Enable CPU offloading: `use_cpu_offload=True`
- Use smaller model: Switch to OmniVideo2-1.3B
- Reduce resolution: Lower width/height
- Reduce frame count: Use fewer frames
- Close other GPU applications
- Reduce batch size if processing multiple videos

#### 2. Slow Generation Speed

**Symptoms**: Generation takes very long (>10 minutes per video)

**Solutions**:
- Disable CPU offloading if you have enough VRAM
- Install flash-attn for faster attention
- Reduce sampling steps (try 30 instead of 40)
- Use faster solver (unipc is fastest)
- Ensure CUDA is properly installed and detected

#### 3. Poor Quality Results

**Symptoms**: Blurry, artifacts, or incorrect edits

**Solutions**:
- Increase sampling steps (40-50)
- Adjust guidance scale (try 3.0-5.0)
- Ensure source video is high quality
- Use more detailed prompts
- Check that all model checkpoints loaded correctly
- Verify T5 and VAE models are correct versions

#### 4. Model Loading Errors

**Symptoms**: Error loading checkpoints or "file not found"

**Solutions**:
- Verify checkpoint directory structure matches expected format
- Check file paths are absolute paths, not relative
- Ensure all checkpoint files are downloaded
- Verify checkpoint file permissions (readable)
- Check that model variant matches available checkpoints

#### 5. Qwen3-VL Errors

**Symptoms**: Vision-language model fails to load or crashes

**Solutions**:
- Ensure Qwen3-VL model is downloaded completely
- Verify transformers library version (4.57.6+)
- Check CUDA compatibility with Qwen3-VL
- Try loading Qwen3-VL separately to isolate issue
- Increase CPU offload delay if timing issues

#### 6. Video Format Compatibility

**Symptoms**: Cannot load certain video formats

**Solutions**:
- Convert videos to MP4 with H.264 codec
- Use ffmpeg to re-encode: `ffmpeg -i input.mov -c:v libx264 output.mp4`
- Ensure decord library is installed correctly
- Check video file is not corrupted

### Debug Mode

Enable detailed logging to diagnose issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Add to the beginning of your node code to see detailed execution logs.

### System Requirements Check

Before starting, verify your system meets requirements:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Additional Resources

### Documentation
- [OmniVideo2 GitHub Repository](https://github.com/SAIS-FUXI/Omni-Video)
- [OmniVideo2 Project Page](https://howellyoung-s.github.io/Omni-Video2-project/)
- [OmniVideo2 Technical Report](https://arxiv.org/abs/2602.08820)
- [ComfyUI Documentation](https://docs.comfy.org/)
- [ComfyUI Custom Node Guide](https://docs.comfy.org/custom-nodes/walkthrough)

### Model Checkpoints
- [OmniVideo2-A14B on HuggingFace](https://huggingface.co/Fudan-FUXI/OmniVideo2-A14B)
- [OmniVideo2-1.3B on HuggingFace](https://huggingface.co/Fudan-FUXI/OmniVideo2-1.3B)
- [Qwen3-VL Model](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)

### Community
- [ComfyUI Discord](https://discord.gg/comfyui)
- [OmniVideo2 Issues](https://github.com/SAIS-FUXI/Omni-Video/issues)

---

## License

This integration guide is provided as-is. Please refer to the original OmniVideo2 and ComfyUI repositories for their respective licenses and usage terms.

---

## Contributing

If you implement this integration or improve upon it, consider:
1. Sharing your implementation on GitHub
2. Contributing back to the OmniVideo2 repository
3. Creating example workflows for the community
4. Writing tutorials or making videos

## Conclusion

Integrating OmniVideo2 into ComfyUI enables powerful video editing capabilities through an intuitive node-based interface. The key challenges are:
- **Memory management**: Requires careful offloading strategies
- **Model integration**: Multiple large models need coordination
- **Video I/O**: Proper handling of video encoding/decoding
- **User experience**: Exposing the right parameters at the right abstraction level

Following this guide, you can create a fully functional ComfyUI integration that makes OmniVideo2's advanced video editing capabilities accessible to a broader audience through ComfyUI's user-friendly interface.
