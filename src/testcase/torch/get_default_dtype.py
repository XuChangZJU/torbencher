import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.get_default_dtype)
class TorchGetUdefaultUdtypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_get_default_dtype_correctness(self):
        initial_dtype = torch.get_default_dtype()  # Get the initial default dtype

        # Check initial default dtype
        assert initial_dtype == torch.float32, f"Initial default dtype is not float32, but {initial_dtype}"

        # List of common floating point dtypes to test
        dtypes_to_test = [torch.float32, torch.float64]

        for dtype in dtypes_to_test:
            torch.set_default_dtype(dtype)
            current_dtype = torch.get_default_dtype()
            assert current_dtype == dtype, f"Default dtype was set to {dtype} but is {current_dtype}"

        # Reset to initial dtype
        torch.set_default_dtype(initial_dtype)
