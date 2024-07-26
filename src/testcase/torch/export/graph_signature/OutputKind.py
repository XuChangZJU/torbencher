import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.graph_signature.OutputKind)
class TorchExportGraphsignatureOutputkindTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_output_kind_correctness(self):
        # Generate random tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        tensor = torch.randn(input_size)

        # Test all possible values of torch.export.graph_signature.OutputKind
        output_kinds = list(torch.export.graph_signature.OutputKind)
        results = []
        for output_kind in output_kinds:
            result = (tensor, output_kind)
            results.append(result)

        return results
