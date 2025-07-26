# © 2025 rndnanthu – Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image

class ColorAnalysisPlotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "plot_type": (["histogram", "parade", "waveform", "vectorscope", "false_color", "gamut_warning"],),
                "exposure": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scope_image",)
    FUNCTION = "analyze"
    CATEGORY = "rndnanthu/🎨Color Tools"

    def analyze(self, image, plot_type="histogram", exposure=1.0):
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError(f"Expected image shape (1, H, W, 3), got {image.shape}")
        # Always keep both formats ready
        img_np = image[0].cpu().numpy()
        img_np = np.clip(img_np * exposure, 0, 1)
        img_torch = torch.clamp(image * exposure, 0, 1)

        if plot_type == "histogram":
            result_np = self.plot_histogram(img_np)
        elif plot_type == "parade":
            result_torch = self.plot_parade(img_torch)
            result_np = result_torch[0].cpu().numpy()
        elif plot_type == "waveform":
            result_torch = self.plot_waveform(img_torch)
            result_np = result_torch[0].cpu().numpy()
        elif plot_type == "vectorscope":
            # 🟢 Use torch directly
            result_torch = self.plot_vectorscope(img_torch)
            result_np = result_torch[0].cpu().numpy()
        elif plot_type == "false_color":
            result_np = self.plot_false_color(img_np)
        elif plot_type == "gamut_warning":
            result_np = self.plot_gamut_warning(img_np)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        return (torch.from_numpy(result_np.astype(np.float32))[None, ...],)

    def plot_histogram(self, img):
        h, w, _ = img.shape
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img.astype(np.float32)], [i], None, [256], [0, 1])
            ax.plot(hist, color=color, linewidth=1)
        ax.set_xlim([0, 256])
        ax.set_title("RGB Histogram")
        ax.axis('off')
        return self._fig_to_img(fig, h, w)

    def plot_parade(self, img_tensor: torch.Tensor) -> torch.Tensor:
        assert img_tensor.ndim == 4 and img_tensor.shape[0] == 1 and img_tensor.shape[3] == 3, \
            f"Expected shape (1, H, W, 3), got {img_tensor.shape}"

        device = img_tensor.device
        B, H, W, C = img_tensor.shape
        scope = torch.zeros((1, H, W * 3, 3), dtype=torch.float32, device=device)

        for ch in range(3):
            channel = img_tensor[0, :, :, ch]
            y_indices = ((1.0 - channel) * (H - 1)).round().long().clamp(0, H - 1)
            x_indices = torch.arange(W, device=device).expand(H, W)
            y_flat = y_indices.flatten()
            x_flat = x_indices.flatten()

            acc = torch.zeros((H, W), dtype=torch.float32, device=device)
            acc.index_put_((y_flat, x_flat), torch.ones_like(y_flat, dtype=torch.float32), accumulate=True)

            col_max = acc.max(dim=0, keepdim=True).values
            col_max[col_max == 0] = 1
            acc = acc / col_max
            acc = acc.pow(0.5)

            scope[0, :, ch * W:(ch + 1) * W, ch] = acc

        return scope

    def plot_waveform(self, img: torch.Tensor) -> torch.Tensor:
        """
        RGB Waveform (torch, CUDA).
        Expects img as (1, H, W, 3) or (H, W, 3) on CUDA, in range [0,1].
        Output: (1, H, W, 3) on CUDA
        """
        if img.dim() == 4 and img.shape[0] == 1:
            img = img[0]  # (H, W, 3)

        H, W, _ = img.shape
        scope = torch.zeros((H, W, 3), device=img.device)

        for c in range(3):
            vals = img[..., c]
            y_pos = ((1.0 - vals) * (H - 1)).long().clamp(0, H - 1)
            x_range = torch.arange(W, device=img.device).repeat(H, 1)
            y_range = y_pos

            scope[y_range, x_range, c] = 1.0

        return scope.unsqueeze(0)

    def plot_vectorscope(self, img: torch.Tensor) -> torch.Tensor:
        """
        Vectorscope plot using U/V chroma mapping.
        Expects img as torch.Tensor (1, H, W, 3) on CUDA, in range [0,1].
        Output: (1, 512, 512, 3) on CUDA.
        """
        img = img.squeeze(0)  # (H, W, 3)
        H, W, _ = img.shape

        rgb_to_yuv = torch.tensor([
            [0.299, -0.14713, 0.615],
            [0.587, -0.28886, -0.51499],
            [0.114, 0.436, -0.10001]
        ], device=img.device)

        yuv = torch.tensordot(img, rgb_to_yuv, dims=([2], [0]))  # (H, W, 3)
        u = yuv[..., 1]
        v = yuv[..., 2]

        px = ((u + 0.5) * 512).clamp(0, 511).long()
        py = ((v + 0.5) * 512).clamp(0, 511).long()

        scope = torch.zeros((512, 512, 3), device=img.device)
        scope[py, px] = torch.tensor([0.0, 1.0, 0.0], device=img.device)  # Green points

        return scope.unsqueeze(0)


    def plot_false_color(self, img):
        gray = np.mean(img, axis=2)
        color = cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        return color[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB

    def plot_gamut_warning(self, img):
        warning = np.zeros_like(img)
        over = np.any(img > 1.0, axis=-1)
        under = np.any(img < 0.0, axis=-1)
        warning[over] = [1, 0, 0]
        warning[under] = [0, 0, 1]
        return np.clip(img + warning * 0.5, 0, 1)

    def _fig_to_img(self, fig, target_h, target_w):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")
        pil_img = pil_img.resize((target_w, target_h))
        plt.close(fig)
        return np.array(pil_img).astype(np.float32) / 255.0

NODE_CLASS_MAPPINGS = {
    "ColorAnalysisPlotNode": ColorAnalysisPlotNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorAnalysisPlotNode": "📈 Color Analysis Scope"
}
