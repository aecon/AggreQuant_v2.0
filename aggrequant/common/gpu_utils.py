"""
GPU configuration utilities.

Author: Athena Economides, 2026, UZH
"""

from aggrequant.common.logging import get_logger

logger = get_logger(__name__)

_tf_memory_growth_configured = False


def configure_tensorflow_memory_growth():
    """
    Enable TensorFlow GPU memory growth so it doesn't pre-allocate all VRAM.

    Safe to call multiple times — the configuration is applied only once.
    Should be called early, before any TF/StarDist model is loaded.
    """
    global _tf_memory_growth_configured
    if _tf_memory_growth_configured:
        return

    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            logger.info(f"TensorFlow memory growth enabled on {len(gpus)} GPU(s)")
    except Exception as e:
        logger.warning(f"Could not configure TensorFlow memory growth: {e}")

    _tf_memory_growth_configured = True
