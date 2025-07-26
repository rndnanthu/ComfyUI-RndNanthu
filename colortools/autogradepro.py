# © 2025 rndnanthu – Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
import numpy as np
import cv2

class AutoGradeProNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "enable_white_balance": ("BOOLEAN", {"default": True}),
            "enable_exposure": ("BOOLEAN", {"default": True}),
            "enable_contrast": ("BOOLEAN", {"default": True}),
            "enable_skintone_preserve": ("BOOLEAN", {"default": True}),
            "enable_hdr_enhance": ("BOOLEAN", {"default": False}),
            "wb_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
            "exposure_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
            "contrast_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
            "skintone_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.01}),
            "hdr_strength": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_image",)
    FUNCTION = "process"
    CATEGORY = "rndnanthu/🎨Color Tools"

    def process(self, image, enable_white_balance, enable_exposure, enable_contrast,
                enable_skintone_preserve, enable_hdr_enhance,
                wb_strength, exposure_strength, contrast_strength,
                skintone_strength, hdr_strength):

        tensor = image[0]

        # Convert to numpy image in [H, W, C]
        if tensor.shape[0] == 3:  # [C, H, W]
            img = tensor.permute(1, 2, 0).cpu().numpy()  # → [H, W, C]
        elif tensor.shape[2] == 3:  # already [H, W, C]
            img = tensor.cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        img = self.ensure_rgb(img)
        original = img.copy()

        if enable_white_balance:
            img = self.white_balance_grayworld(img, wb_strength)

        if enable_exposure:
            img = self.auto_exposure(img, exposure_strength)

        if enable_contrast:
            img = self.auto_contrast(img, contrast_strength)

        if enable_skintone_preserve:
            img = self.preserve_skintones(original, img, skintone_strength)

        if enable_hdr_enhance:
            img = self.hdr_enhance(original, img, hdr_strength)

        # 🧠 Output in channel-last format: [1, H, W, C]
        out = torch.from_numpy(np.clip(img, 0, 1)).unsqueeze(0).contiguous()

        return (out,)



    # === 🔧 Safety First — Enforce RGB Shape ===

    def ensure_rgb(self, img):
        """Force (H, W, 3) RGB format, regardless of grayscale, 1-channel, etc."""
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] != 3:
                raise ValueError(f"Invalid channel count: {img.shape}")
        else:
            raise ValueError(f"Image has invalid dimensions: {img.shape}")
        return np.clip(img, 0, 1).astype(np.float32)


    # === 🎨 Correction Functions ===

    def white_balance_grayworld(self, img, strength):
        img = self.ensure_rgb(img)
        avg = np.mean(img, axis=(0, 1))
        gray = np.mean(avg)
        gain = gray / (avg + 1e-6)
        balanced = img * gain
        return img * (1 - strength) + balanced * strength

    def auto_exposure(self, img, strength):
        img = self.ensure_rgb(img)

        if img.ndim != 3 or img.shape[2] != 3:
            print(f"[ERROR] auto_exposure: image shape is invalid: {img.shape}")
            raise ValueError(f"Image must be RGB with 3 channels, got shape {img.shape}")

        # Force correct shape before OpenCV conversion
        rgb_uint8 = (img * 255).astype(np.uint8)
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            rgb_uint8 = np.stack([rgb_uint8] * 3, axis=-1)

        print(f"[DEBUG] rgb_uint8 shape: {rgb_uint8.shape}")  # Must be (H, W, 3)

        # Convert to HSV
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)

        # Work on the V channel
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        v_min, v_max = np.percentile(v, [2, 98])
        stretched = np.clip((v - v_min) / (v_max - v_min + 1e-6), 0, 1)
        hsv[:, :, 2] = ((1 - strength) * v + strength * stretched) * 255

        out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out_rgb.astype(np.float32) / 255.0


    def auto_contrast(self, img, strength):
        img = self.ensure_rgb(img)
        yuv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
        y = yuv[:, :, 0].astype(np.float32) / 255.0
        mean = y.mean()
        stretched = 0.5 * (1 + np.tanh(2 * (y - mean)))
        yuv[:, :, 0] = ((1 - strength) * y + strength * stretched) * 255
        return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0

    def preserve_skintones(self, original, corrected, strength):
        original = self.ensure_rgb(original)
        corrected = self.ensure_rgb(corrected)

        # Convert to HSV
        hsv = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

        # Define skin tone range (in OpenCV HSV: H=0~180, S=0~255, V=0~255)
        lower = np.array([0, 30, 60], dtype=np.uint8)     # reddish hue, min saturation and brightness
        upper = np.array([35, 180, 255], dtype=np.uint8)  # yellow-orange hue, max saturation

        skin_mask = cv2.inRange(hsv, lower, upper).astype(np.float32) / 255.0  # [0.0, 1.0] float mask

        # Smooth the mask to avoid hard edges
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)

        # Expand mask to match shape [H, W, 3]
        mask_3c = np.repeat(skin_mask[:, :, np.newaxis], 3, axis=2)

        # Blend corrected and original using the skin mask
        blended = corrected * (1 - mask_3c * strength) + original * (mask_3c * strength)
        return blended


    def hdr_enhance(self, original, corrected, strength):
        corrected = self.ensure_rgb(corrected)
        enhanced = cv2.detailEnhance((corrected * 255).astype(np.uint8), sigma_s=10, sigma_r=0.15)
        enhanced = enhanced.astype(np.float32) / 255.0
        return corrected * (1 - strength) + enhanced * strength


# === Bind to ComfyUI ===

NODE_CLASS_MAPPINGS = {
    "AutoGradePro": AutoGradeProNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoGradePro": "🎨 AutoGradePro 2.0"
}
