"""Import torch/lpips before sklearn: on some Windows installs the reverse order segfaults (DLL clash)."""
import lpips  # noqa: F401
import sklearn.decomposition  # noqa: F401
