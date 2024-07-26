import torch
import random
import io
import pathlib

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.load)
class TorchExportLoadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_load_correctness(self):
        # Generate a random tensor to simulate an ExportedProgram
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        tensor_data = torch.randn(input_size)

        # Save the tensor data to a file-like object
        buffer = io.BytesIO()
        torch.save(tensor_data, buffer)
        buffer.seek(0)

        # Load the ExportedProgram from the file-like object
        loaded_tensor_data = torch.load(buffer)

        return loaded_tensor_data
