# © 2025 rndnanthu – Licensed under CC BY-NC 4.0
import torch
import numpy as np
import cv2

class ProColorGrading:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "midtones": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "levels_in_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "levels_in_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "levels_out_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "levels_out_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graded_image",)
    FUNCTION = "grade"
    CATEGORY = "rndnanthu/🎨Color Tools"

    def grade(self, image, shadows, midtones, highlights,
              exposure, hue_shift, saturation, vibrance,
              contrast, pivot, color_boost, temperature, tint,
              levels_in_black, levels_in_white, levels_out_black, levels_out_white):

        img = image[0].cpu().numpy().astype(np.float32)
        img = np.clip(img, 0, 1)

        # Exposure
        if exposure != 0.0:
            img *= 2.0 ** exposure
            img = np.clip(img, 0, 1)

        # Input levels
        if levels_in_black != 0.0 or levels_in_white != 1.0:
            img = (img - levels_in_black) / max(1e-5, levels_in_white - levels_in_black)
            img = np.clip(img, 0, 1)

        # Tone mask adjustments
        if shadows != 0.0 or midtones != 0.0 or highlights != 0.0:
            lum = np.mean(img, axis=2, keepdims=True)
            shadows_mask = np.clip(1.0 - lum * 2.0, 0.0, 1.0)
            highlights_mask = np.clip((lum - 0.5) * 2.0, 0.0, 1.0)
            midtones_mask = 1.0 - shadows_mask - highlights_mask

            img += shadows * shadows_mask
            img += midtones * midtones_mask
            img += highlights * highlights_mask
            img = np.clip(img, 0, 1)

        # Temperature/tint
        if temperature != 0.0 or tint != 0.0:
            wb = np.array([
                1.0 + temperature * 0.1 - tint * 0.05,
                1.0,
                1.0 - temperature * 0.1 + tint * 0.05
            ], dtype=np.float32).reshape((1, 1, 3))
            img *= wb
            img = np.clip(img, 0, 1)

        # HSV adjustments: only apply if needed
        if hue_shift != 0.0 or saturation != 1.0 or vibrance != 0.0:
            img_255 = (img * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_255, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            if hue_shift != 0.0:
                h = (h.astype(np.int32) + int(hue_shift)) % 180
                h = h.astype(np.uint8)

            if saturation != 1.0:
                s = np.clip(s.astype(np.float32) * saturation, 0, 255).astype(np.uint8)

            if vibrance > 0.0:
                mean_sat = np.mean(s)
                mask = s < mean_sat
                s[mask] = np.clip(s[mask] + vibrance * (255 - s[mask]), 0, 255).astype(np.uint8)

            hsv_mod = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            img = np.clip(img, 0, 1)

        # Contrast
        if contrast != 1.0:
            img = (img - pivot) * contrast + pivot
            img = np.clip(img, 0, 1)

        # Color boost
        if color_boost != 1.0:
            avg = np.mean(img, axis=2, keepdims=True)
            img = avg + (img - avg) * color_boost
            img = np.clip(img, 0, 1)

        # Output levels
        if levels_out_black != 0.0 or levels_out_white != 1.0:
            img = img * (levels_out_white - levels_out_black) + levels_out_black
            img = np.clip(img, 0, 1)

        return (torch.from_numpy(img).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "ProColorGrading": ProColorGrading,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProColorGrading": "🎛️ Pro Color Grading",
}
