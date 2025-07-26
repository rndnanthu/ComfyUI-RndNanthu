# © 2025 rndnanthu – Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
import numpy as np
import cv2
import os
import json

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRESETS_FILE = os.path.join(BASE_PATH, "assets", "presets", "film_grain_presets.json")

class FilmGrainNode:
    @classmethod
    def INPUT_TYPES(cls):
        presets = []
        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, "r") as f:
                    presets = list(json.load(f).keys())
            except Exception as e:
                print("Error loading presets:", e)
        return {
            "required": {
                "image": ("IMAGE",),
                "preset_name": (["None"] + presets, {"default": "None"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01} ),
                "grain_size": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.01}),
                "color_var": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grained",)
    FUNCTION = "apply_grain"
    CATEGORY = "rndnanthu/🎨Grain Tools"

    def __init__(self):
        self.presets = {}
        self.load_presets()

    def load_presets(self):
        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, "r") as f:
                    self.presets = json.load(f)
            except Exception as e:
                print("Failed to load presets:", e)
                self.presets = {}

    def apply_grain(self, image, preset_name, intensity, strength, grain_size, color_var, sharpen):
            img = image[0].cpu().numpy()  # (H, W, C)

            # Apply preset if selected
            if preset_name and preset_name in self.presets:
                p = self.presets[preset_name]
                strength = p.get("strength", strength)
                grain_size = p.get("grain_size", grain_size)
                color_var = p.get("color_var", color_var)
                sharpen = p.get("sharpen", sharpen)

            # Apply intensity globally
            strength *= intensity
            color_var *= intensity
            sharpen *= intensity

            h, w, c = img.shape  # (H, W, 3)

            # === Generate monochromatic noise ===
            mono_noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
            mono_noise = cv2.GaussianBlur(mono_noise, (0, 0), grain_size)
            mono_noise = ((mono_noise - mono_noise.min()) / (mono_noise.max() - mono_noise.min() + 1e-8)) * 2 - 1
            mono_noise = np.repeat(mono_noise[:, :, np.newaxis], 3, axis=2)

            # === Generate chromatic noise ===
            chroma_noise = np.random.normal(0, 1, (h, w, 3)).astype(np.float32)
            for ch in range(3):
                chroma_noise[:, :, ch] = cv2.GaussianBlur(chroma_noise[:, :, ch], (0, 0), grain_size)
            chroma_noise = ((chroma_noise - chroma_noise.min()) / (chroma_noise.max() - chroma_noise.min() + 1e-8)) * 2 - 1

            # === Combine noise ===
            grain = (1.0 - color_var) * mono_noise + color_var * chroma_noise
            grained = img + grain * strength
            grained = np.clip(grained, 0.0, 1.0)

            # === Optional sharpening ===
            if sharpen > 0.0:
                blur = cv2.GaussianBlur(grained, (0, 0), 1.0)
                grained = cv2.addWeighted(grained, 1 + sharpen, blur, -sharpen, 0)

            output = torch.from_numpy(grained).unsqueeze(0).float()  # (1, H, W, 3)
            return (output,)



NODE_CLASS_MAPPINGS = {
    "FilmGrain": FilmGrainNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilmGrain": "🎞 Film Grain (Advanced)"
}
