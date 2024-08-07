import torch
import torch.nn.functional as F

from src import TorchWrapper

wrapper = TorchWrapper(
    {
        "out_dir": "result",
        "format": "csv",
        "file_max_size": "512MB",
        "file_name_spec": "timestamp",
    }
)


def my_code(*args, **kwargs):
    """
    此处写需要运行的程序
    """
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = F.relu(a + b)
    return c


result = wrapper.start(my_code, 1, 2, x=3, y=4)
print(result)