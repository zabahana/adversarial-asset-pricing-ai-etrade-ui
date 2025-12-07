"""Lightning Works module with fallback support when Lightning framework is unavailable."""

# Try to import LightningWork, fallback to simple base class
try:
    from lightning.app import LightningWork
    HAS_LIGHTNING = True
except ImportError:
    try:
        from lightning import LightningWork
        HAS_LIGHTNING = True
    except ImportError:
        # Fallback: create a simple base class that mimics LightningWork
        class LightningWork:
            """Fallback base class when Lightning framework is not available."""
            def __init__(self, *args, **kwargs):
                # Accept any kwargs but do nothing
                pass
        HAS_LIGHTNING = False

__all__ = ['LightningWork', 'HAS_LIGHTNING']

