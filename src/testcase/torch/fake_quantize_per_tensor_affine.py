import torch
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fake_quantize_per_tensor_affine)
class TorchFakeUquantizeUperUtensorUaffineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_fake_quantize_per_tensor_affine_correctness(self):
        x = torch.randn(4)

        # Ensure scale and zero_point are on the same device as x
        scale = torch.tensor(0.1, device=x.device)
        zero_point = torch.tensor(0, device=x.device, dtype=torch.int)

        return (
            x,
            torch.fake_quantize_per_tensor_affine(x, 0.1, 0, 0, 255),  # Scalars 0.1 and 0 are automatically on CPU
            torch.fake_quantize_per_tensor_affine(x, scale, zero_point, 0, 255)
        )
