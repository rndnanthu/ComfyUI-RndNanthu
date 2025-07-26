# © 2025 rndnanthu – Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
import numpy as np
import os
import glob

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LUT_FOLDER = os.path.join(BASE_PATH, "assets", "luts")

class ColorSpaceSimNode:
    @classmethod
    def INPUT_TYPES(cls):
        lut_files = glob.glob(os.path.join(LUT_FOLDER, "*.cube"))
        lut_names = [os.path.basename(f) for f in lut_files]
        lut_map = dict(zip(lut_names, lut_files))
        cls._lut_map = lut_map

        return {
            "required": {
                "image": ("IMAGE",),
                "target_profile": (
                    ["Sony S-Log3", "Arri LogC", "Canon C-Log", "Rec.709"],
                    {"default": "Sony S-Log3"}
                ),
                "simulate_camera_encode": ("BOOLEAN", {"default": False}),
                "enable_gamma": ("BOOLEAN", {"default": False}),
                "gamma_value": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.01}),
                "enable_lut": ("BOOLEAN", {"default": False}),
                "lut_name": (
                    ["None"] + sorted(lut_names) if lut_names else ["None"],
                    {"default": "None"}
                ),
                "profile_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_tone_map": ("BOOLEAN", {"default": True}),
                "contrast_boost": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out",)
    FUNCTION = "run"
    CATEGORY = "rndnanthu/🎨Color Tools"

    def run(self, image, target_profile, simulate_camera_encode,
            enable_gamma, gamma_value, enable_lut, lut_name,
            profile_strength, enable_tone_map, contrast_boost):

        img = image  # shape (1, H, W, 3)
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"Expected image shape (1, H, W, 3), got {img.shape}")

        img = img.clamp(0, 1).float()
        original = img.clone()

        if simulate_camera_encode and target_profile != "Rec.709":
            img = self.srgb_to_linear_tensor(img)

        if target_profile == "Sony S-Log3":
            img = self.slog3_curve_tensor(img)
        elif target_profile == "Arri LogC":
            img = self.logc_curve_tensor(img)
        elif target_profile == "Canon C-Log":
            img = self.clog_curve_tensor(img)

        if enable_tone_map:
            img = self.simple_tone_map_tensor(img)

        if enable_gamma:
            img = torch.pow(img.clamp(0, 1), 1.0 / gamma_value)

        if enable_lut and lut_name in self._lut_map:
            try:
                img = self.apply_cube_lut_gpu(img, self._lut_map[lut_name])
            except Exception as e:
                print("⚠️ LUT failed:", e)

        if contrast_boost != 1.0:
            img = self.boost_contrast_tensor(img, contrast_boost)

        final = torch.clamp(original * (1.0 - profile_strength) + img * profile_strength, 0, 1)
        return (final,)

    def srgb_to_linear_tensor(self, img):
        return torch.where(
            img <= 0.04045, img / 12.92,
            torch.pow((img + 0.055) / 1.055, 2.4)
        )

    def slog3_curve_tensor(self, x):
        a, b = 0.432699, 0.037584
        return torch.where(
            x < 0.011,
            0.092864 * x + 0.00404,
            a * torch.log10(x + b) + 0.616596
        )

    def logc_curve_tensor(self, x):
        a, b, c, d, e, f, cut = 5.555556, 0.052272, 0.24719, 0.385537, 5.367655, 0.092809, 0.010591
        return torch.where(
            x > cut,
            c * torch.log10(a * x + b) + d,
            e * x + f
        )

    def clog_curve_tensor(self, x):
        return torch.log10(500 * x + 1) / np.log10(501)

    def simple_tone_map_tensor(self, x):
        return x / (x + 0.25)

    def boost_contrast_tensor(self, x, boost):
        mid = 0.5
        return torch.clamp((x - mid) * boost + mid, 0, 1)

    def load_cube_lut_torch(self, path):
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        size_line = [l for l in lines if l.startswith('LUT_3D_SIZE')]
        if not size_line:
            raise ValueError("Missing LUT_3D_SIZE in cube file")

        size = int(size_line[0].split()[1])

        data = [list(map(float, l.split())) for l in lines if len(l.split()) == 3]
        if len(data) != size ** 3:
            raise ValueError(f"Invalid LUT size: expected {size**3}, got {len(data)}")

        lut_np = np.array(data, dtype=np.float32).reshape((size, size, size, 3))  # (R, G, B, 3)
        lut_torch = torch.from_numpy(lut_np).permute(3, 0, 1, 2).contiguous()  # (3, R, G, B)
        return lut_torch, size

    def apply_cube_lut_gpu(self, img, path):
        lut, size = self.load_cube_lut_torch(path)

        if not torch.is_tensor(img) or img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError("Expected image shape (1, H, W, 3)")

        device = img.device
        B, H, W, C = img.shape

        # normalize image to [-1, 1]
        norm = img * 2 - 1
        grid = norm.view(B, H, W, 1, C)  # (B, H, W, 1, 3)

        lut = lut.to(device).unsqueeze(0)  # shape: (1, 3, size, size, size)

        sampled = torch.nn.functional.grid_sample(
            lut, grid, mode='bilinear', align_corners=True
        )  # returns (1, 3, H, W, 1)

        out = sampled.squeeze(-1).permute(0, 2, 3, 1)  # -> (1, H, W, 3)
        return out.clamp(0, 1)


NODE_CLASS_MAPPINGS = {
    "ColorSpaceSim": ColorSpaceSimNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorSpaceSim": "🎥 Color Space Simulator"
}
