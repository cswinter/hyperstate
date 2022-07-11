from typing import Any, Callable, Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def encode(obj: Any, chain: Optional[Callable[[Any], Any]] = None) -> Any:
    tensor_type = None
    if torch is not None and isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
        tensor_type = "torch"

    if isinstance(obj, np.ndarray):
        return {
            b"__tensor__": tensor_type or "numpy",
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
            b"data": np.ndarray.tobytes(obj),
        }
    else:
        return obj if chain is None else chain(obj)


def decode(obj: Any, chain: Optional[Callable[[Any], Any]] = None) -> Any:
    try:
        if b"__tensor__" in obj:
            tensor: Any = np.ndarray(
                buffer=obj[b"data"],
                dtype=np.dtype(obj[b"dtype"]),
                shape=obj[b"shape"],
            )
            if obj[b"__tensor__"] == "torch" and torch is not None:
                return torch.from_numpy(tensor)
            else:
                return tensor
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)
