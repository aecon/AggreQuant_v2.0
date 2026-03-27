"""StarDist-based nuclei segmentation."""

import numpy as np
import skimage.filters
import skimage.morphology
from aggrequant.segmentation.base import BaseSegmenter


# Default parameters
SIGMA_DENOISE = 2
SIGMA_BACKGROUND = 50
MIN_NUCLEUS_AREA = 300
MAX_NUCLEUS_AREA = 15000


class StarDistSegmenter(BaseSegmenter):
    """
    Nuclei segmentation using StarDist 2D.

    StarDist is a deep learning model for star-convex object detection,
    particularly suited for nuclei segmentation in fluorescence microscopy.

    The pipeline:
    1. Pre-processing: Gaussian denoise + background normalization
    2. Inference: StarDist predict_instances
    3. Post-processing: Size exclusion, border separation
    """

    def __init__(
        self,
        model_name: str = "2D_versatile_fluo",
        sigma_denoise: float = SIGMA_DENOISE,
        sigma_background: float = SIGMA_BACKGROUND,
        min_area: int = MIN_NUCLEUS_AREA,
        max_area: int = MAX_NUCLEUS_AREA,
        verbose: bool = False,
    ):
        """
        Initialize StarDist segmenter.

        Arguments:
            model_name: Pre-trained model name (default: '2D_versatile_fluo')
            sigma_denoise: Gaussian sigma for denoising
            sigma_background: Gaussian sigma for background estimation
            min_area: Minimum nucleus area in pixels
            max_area: Maximum nucleus area in pixels
            verbose: Print progress messages
        """
        super().__init__(verbose=verbose)

        self.model_name = model_name
        self.sigma_denoise = sigma_denoise
        self.sigma_background = sigma_background
        self.min_area = min_area
        self.max_area = max_area

        self._model = None

    @property
    def name(self) -> str:
        return "StarDistSegmenter"

    @property
    def model(self):
        """Lazy load the StarDist model."""
        if self._model is None:
            self._log(f"Loading StarDist model: {self.model_name}")
            from stardist.models import StarDist2D
            self._model = StarDist2D.from_pretrained(self.model_name)
        return self._model

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment nuclei in the input image.

        Arguments:
            image: Input grayscale image (2D array)

        Returns:
            labels: Instance segmentation labels (uint16)
                    0 = background, 1+ = individual nuclei
        """
        # Pre-processing
        preprocessed = self._preprocess(image)

        # StarDist inference
        labels = self._segment_stardist(preprocessed)

        # Post-processing
        labels_ = self._postprocess_size_exclusion(labels)
        labels_ = self._postprocess_increase_borders(labels_)

        # Relabel to ensure consecutive labels, using a LUT (fast O(N), no re-labeling)
        #labels = skimage.morphology.label(labels > 0).astype(np.uint16)
        unique = np.unique(labels_)  # sorted unique values including 0 (background)
        lut = np.zeros(int(unique.max()) + 1, dtype=np.uint16)
        lut[unique] = np.arange(len(unique), dtype=np.uint16)
        labels = lut[labels_]

        self._log(f"Final count: {labels.max()} nuclei")
        return labels

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Pre-process image for StarDist.

        Applies Gaussian denoising and background normalization.
        Output: I_normalized = Gaussian(sigma_denoise) / Gaussian(sigma_background)
        """
        # Convert to float
        img = image.astype(np.float32)

        # Denoise
        denoised = skimage.filters.gaussian(
            img, sigma=self.sigma_denoise, mode='reflect', preserve_range=True
        )

        # Background estimation
        background = skimage.filters.gaussian(
            denoised, sigma=self.sigma_background, mode='reflect', preserve_range=True
        )

        # Normalize
        normalized = denoised / (background + 1e-8)

        return normalized

    def _segment_stardist(self, image: np.ndarray) -> np.ndarray:
        """Run StarDist inference."""
        from csbdeep.utils import normalize

        labels, _ = self.model.predict_instances(
            normalize(image), # normalizes to [0,1] range using percentiles.
            predict_kwargs=dict(verbose=False)
        )
        return labels

    def _postprocess_size_exclusion(self, labels: np.ndarray) -> np.ndarray:
        """Remove nuclei outside the allowed size range."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        removed_small = 0
        removed_large = 0

        for label_id, area in zip(unique_labels[1:], counts[1:]):  # Skip background (0)
            if area < self.min_area:
                labels[labels == label_id] = 0
                removed_small += 1
            elif area > self.max_area:
                labels[labels == label_id] = 0
                removed_large += 1

        return labels

    def _postprocess_increase_borders(self, labels: np.ndarray) -> np.ndarray:
        """Add separation between touching nuclei."""
        # Find edges using Sobel filter
        edges = skimage.filters.sobel(labels)
        edge_mask = edges > 0

        # Dilate edges
        fat_edges = skimage.morphology.dilation(edge_mask)

        # Create output with borders set to 0
        objects = labels.copy()
        objects[fat_edges] = 0

        if len(np.unique(objects)) != len(np.unique(labels)):
            raise ValueError(
                f"Border thickening eroded "
                f"{len(np.unique(labels)) - len(np.unique(objects))} nuclei"
            )

        return objects
