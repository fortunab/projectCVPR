"""
Handcrafted Texture & Color Feature Extractor for Colonoscopy Mucosal Surfaces (PolypGen).
Extracts:
1. Multiscale LBP (Uniform rotation-invariant patterns)
2. GLCM (Haralick properties on mucosal surface)
3. Gabor Filter Bank (Multidirectional frequencies)
4. Color Statistics & Histograms (RGB, HSV, CIE-Lab)
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from scipy import stats, ndimage
from skimage import color
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


class PolypFeatureExtractor:
    def __init__(
        self,
        lbp_radii_points: List[Tuple[int, int]] = [(1, 8), (2, 16), (3, 24)],
        glcm_distances: List[int] = [1, 2, 3, 5],
        glcm_angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
        gabor_frequencies: List[float] = [0.05, 0.1, 0.2, 0.35],
        gabor_orientations: List[float] = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    ):
        self.lbp_radii_points = lbp_radii_points
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        self.gabor_frequencies = gabor_frequencies
        self.gabor_orientations = gabor_orientations

        self.gabor_kernels = []
        for freq in self.gabor_frequencies:
            for theta in self.gabor_orientations:
                self.gabor_kernels.append((
                    np.real(gabor_kernel(freq, theta=theta)),
                    np.imag(gabor_kernel(freq, theta=theta))
                ))

    def extract_single_image(self, rgb_img: np.ndarray) -> np.ndarray:
        # Resize to standard patch resolution if large
        if rgb_img.shape[0] > 224 or rgb_img.shape[1] > 224:
            pil_img = Image.fromarray(rgb_img).resize((224, 224), Image.Resampling.BICUBIC)
            rgb_img = np.array(pil_img)

        gray_img = (color.rgb2gray(rgb_img) * 255.0).astype(np.uint8)
        blocks = []

        # 1. LBP (54 dims)
        for r, p in self.lbp_radii_points:
            lbp = local_binary_pattern(gray_img, P=p, R=r, method='uniform')
            n_bins = p + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            blocks.append(hist)

        # 2. GLCM (144 dims)
        gray_q = (gray_img // 4).astype(np.uint8)
        glcm = graycomatrix(gray_q, distances=self.glcm_distances, angles=self.glcm_angles, levels=64, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            val = graycoprops(glcm, prop)
            blocks.append(val.ravel())
            blocks.append(np.mean(val, axis=1))
            blocks.append(np.std(val, axis=1))

        # 3. Gabor Filter Bank (72 dims)
        img_f = gray_img.astype(np.float64) / 255.0
        gabor_feats = []
        for kr, ki in self.gabor_kernels:
            fr = ndimage.convolve(img_f, kr, mode='reflect')
            fi = ndimage.convolve(img_f, ki, mode='reflect')
            mag = np.hypot(fr, fi)
            gabor_feats.extend([np.mean(mag), np.std(mag), np.mean(mag**2)])
        blocks.append(np.array(gabor_feats, dtype=np.float64))

        # 4. Color Statistics & Histograms (75 dims)
        color_feats = []
        rgb_f = rgb_img.astype(np.float64) / 255.0
        for c in range(3):
            ch = rgb_f[:, :, c].ravel()
            color_feats.extend([np.mean(ch), np.std(ch), stats.skew(ch)])

        hsv_img = color.rgb2hsv(rgb_img)
        for c in range(3):
            ch = hsv_img[:, :, c].ravel()
            color_feats.extend([np.mean(ch), np.std(ch), stats.skew(ch)])
        h_hist, _ = np.histogram(hsv_img[:, :, 0].ravel(), bins=16, range=(0, 1), density=True)
        s_hist, _ = np.histogram(hsv_img[:, :, 1].ravel(), bins=8, range=(0, 1), density=True)
        color_feats.extend(h_hist)
        color_feats.extend(s_hist)

        lab_img = color.rgb2lab(rgb_img)
        for c in range(3):
            ch = lab_img[:, :, c].ravel()
            color_feats.extend([np.mean(ch), np.std(ch), stats.skew(ch)])

        for c in range(3):
            ch_hist, _ = np.histogram(rgb_f[:, :, c].ravel(), bins=8, range=(0, 1), density=True)
            color_feats.extend(ch_hist)

        blocks.append(np.array(color_feats, dtype=np.float64))
        return np.concatenate(blocks).astype(np.float64)


def extract_polyp_features_parallel(
    image_paths: List[str],
    labels: np.ndarray,
    n_jobs: int = -1,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            print(f"[FeatureExtractor] Loading cached features from {cache_path}...")
            data = np.load(cache_path)
            return data["X"], data["y"]

    extractor = PolypFeatureExtractor()
    print(f"[FeatureExtractor] Extracting handcrafted features on {len(image_paths)} colonoscopy images...")

    def process(p):
        img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        return extractor.extract_single_image(img)

    results = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(process)(p) for p in tqdm(image_paths, desc="Extracting Mucosal Descriptors")
    )
    X = np.nan_to_num(np.vstack(results), nan=0.0)
    y = np.array(labels, dtype=np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y
