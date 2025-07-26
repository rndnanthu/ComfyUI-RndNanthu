# © 2025 rndnanthu – Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
import os
import torch
import numpy as np
import cv2

# Optional EXR support
try:
    import OpenEXR, Imath
    exr_available = True
except ImportError:
    exr_available = False


class ConvertToLogImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "log_type": (["cineon", "logc", "redlog"], {"default": "cineon"}),
                "is_srgb": ("BOOLEAN", {"default": True}),
                "tone_map_hdr": ("BOOLEAN", {"default": False}),
                "enhance_bit_depth": ("BOOLEAN", {"default": False}),
                "save_as_openexr": ("BOOLEAN", {"default": False}),
                "exr_filename": ("STRING", {"default": "log_output.exr"}),
            },
            "optional": {
                "depth_map": ("IMAGE",),
                "gamma_value": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.8, "min": 0.1, "max": 5.0, "step": 0.01}),
                "midpoint": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_boost": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("log_image",)
    FUNCTION = "run"
    CATEGORY = "rndnanthu/🎨Color Tools"

    # === Color Conversion ===

    def srgb_to_linear(self, c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def rec709_to_linear(self, c):
        return np.where(c < 0.081, c / 4.5, ((c + 0.099) / 1.099) ** (1 / 0.45))

    def tone_map(self, l):
        return l / (l + 1.0)

    def linear_to_cineon(self, l):
        l = np.clip(l, 1e-6, 1)
        return (np.log10(l * 685 + 95) - np.log10(95)) / (np.log10(780) - np.log10(95))

    def linear_to_logc(self, l):
        a, b, c_, d, e, f, cut = 5.555556, 0.052272, 0.2471896, 0.385537, 5.367655, 0.092809, 0.010591
        l = np.clip(l, 1e-6, 1)
        return np.where(l > cut, c_ * np.log10(a * l + b) + d, e * l + f)

    def linear_to_redlog(self, l):
        l = np.clip(l, 1e-6, 1)
        return (np.log10(l + 0.01) + 0.6) / 1.6

    def convert_to_log(self, l, log_type):
        funcs = {
            "cineon": self.linear_to_cineon,
            "logc": self.linear_to_logc,
            "redlog": self.linear_to_redlog
        }
        return funcs[log_type](l)

    # === Dynamic Range Enhancement ===

    def enhance_dynamic_range(self, img, gamma_value, contrast, midpoint, depth_map=None, depth_boost=0.5):
        img = np.clip(img, 0.001, 1.0)

        # === Step 1: Depth boost applied FIRST in linear space ===
        if depth_map is not None:
            dmap = depth_map.squeeze().cpu().numpy().astype(np.float32)

            # If 3-channel, convert to grayscale
            if dmap.ndim == 3 and dmap.shape[-1] == 3:
                dmap = cv2.cvtColor(dmap, cv2.COLOR_RGB2GRAY)

            # Resize & blur
            dmap = cv2.resize(dmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            dmap = cv2.GaussianBlur(dmap, (5, 5), 0)
            dmap = np.clip(dmap, 0.0, 1.0)

            # Expand to RGB shape
            dmap = np.expand_dims(dmap, axis=-1)
            dmap = np.repeat(dmap, 3, axis=2)

            # Apply depth boost
            img += dmap * depth_boost
            img = np.clip(img, 0.001, 1.0)  # Re-clip after depth influence

        # === Step 2: Gamma correction ===
        img_gamma = np.power(img, gamma_value)

        # === Step 3: Sigmoid contrast ===
        img_contrast = 1 / (1 + np.exp(-contrast * (img_gamma - midpoint)))

        return np.clip(img_contrast, 0, 1)


    # === EXR Export ===

    def export_exr(self, img_np, path):
        if not exr_available:
            print("[Log Converter] OpenEXR not available.")
            return
        if img_np.shape[-1] != 3:
            raise ValueError("EXR export requires 3-channel RGB image.")
        h, w, _ = img_np.shape
        header = OpenEXR.Header(w, h)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        header['channels'] = {'R': Imath.Channel(pt), 'G': Imath.Channel(pt), 'B': Imath.Channel(pt)}
        exr = OpenEXR.OutputFile(path, header)
        r, g, b = [img_np[:, :, i].astype(np.float32).tobytes() for i in range(3)]
        exr.writePixels({'R': r, 'G': g, 'B': b})
        exr.close()

    # === Main ===

    def run(
        self, image, log_type, is_srgb, tone_map_hdr, enhance_bit_depth,
        save_as_openexr, exr_filename, depth_map=None,
        gamma_value=0.8, contrast=1.2, midpoint=0.5, depth_boost=0.5
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"[Log Converter] Expected shape [B,H,W,3], got: {image.shape}")

        img = image[0].cpu().numpy()  # [H,W,3]
        img = np.clip(img, 0, 1).astype(np.float32)

        # Convert to linear
        linear = self.srgb_to_linear(img) if is_srgb else self.rec709_to_linear(img)

        # Tone map if needed
        if tone_map_hdr:
            linear = self.tone_map(linear)

        # Convert to log
        log_img = self.convert_to_log(linear, log_type)
        log_img = np.clip(log_img, 0, 1)

        # Bit-depth enhancement
        if enhance_bit_depth:
            log_img = self.enhance_dynamic_range(
                log_img, gamma_value, contrast, midpoint, depth_map, depth_boost
            )

        # Save as EXR
        if save_as_openexr:
            os.makedirs("output", exist_ok=True)
            self.export_exr(log_img, os.path.join("output", exr_filename))

        output_tensor = torch.from_numpy(log_img).unsqueeze(0).contiguous().float()
        return (output_tensor,)


# === ComfyUI Registration ===

NODE_CLASS_MAPPINGS = {
    "ConvertToLogImage": ConvertToLogImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConvertToLogImage": "🎚️ Log Converter (Cineon/LogC/REDLog)"
}
