import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.testing.assert_allclose)
class TorchTestingAssertUallcloseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_assert_allclose_correctness(self):
        """Test correctness with small scale random parameters."""
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        actual = torch.randn(input_size)
        expected = actual.clone()  # Ensure tensors are close

        # Call assert_allclose to check for closeness
        result = torch.testing.assert_allclose(actual, expected)

        return result
