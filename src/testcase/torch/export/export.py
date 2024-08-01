import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.export)
class TorchExportExportTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_correctness(self):
        class SimpleNet(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.concat([x, y])

        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensors for inputs
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        net = SimpleNet()

        # Export the function with the random tensors as example inputs
        exported_program = torch.export.export(net, (tensor1, tensor2))

        return exported_program
