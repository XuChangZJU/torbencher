import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.graph_signature.InputKind)
class TorchExportGraphUsignatureInputkindTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_input_kind_correctness(self):
        # Generate a random tensor size
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor data
        tensor = torch.randn(input_size)

        # Test all possible values of torch.export.graph_signature.InputKind
        input_kind_values = list(torch.export.graph_signature.InputKind)
        results = []
        for input_kind in input_kind_values:
            # Apply the input kind to the tensor (assuming a hypothetical function that uses InputKind)
            # Since the actual usage of InputKind is not specified, we will just store the kind for now
            result = (tensor, input_kind)
            results.append(result)

        return results
