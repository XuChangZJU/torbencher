import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.set_default_dtype)
class TorchSetdefaultdtypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_default_dtype_correctness(self):
        # Randomly chose either torch.float32 or torch.float64 for dtype
        dtype = random.choice([torch.float32, torch.float64])

        # Set the chosen dtype as the default dtype
        torch.set_default_dtype(dtype)

        # Create a float tensor and check its dtype
        tensor_floats = torch.tensor([random.uniform(0.1, 10.0) for _ in range(random.randint(1, 5))])
        dtype_floats_result = tensor_floats.dtype

        # Create a complex tensor and check its dtype
        tensor_complex = torch.tensor(
            [complex(random.uniform(0.1, 10.0), random.uniform(0.1, 10.0)) for _ in range(random.randint(1, 5))])
        dtype_complex_result = tensor_complex.dtype

        return dtype, dtype_floats_result, dtype_complex_result
