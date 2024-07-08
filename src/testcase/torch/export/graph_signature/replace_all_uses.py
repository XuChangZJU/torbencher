import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.export.graph_signature.replace_all_uses)
class TorchExportGraphsignatureReplaceallusesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_replace_all_uses_correctness(self):
        # Randomly generate the size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Create a graph signature
        graph_signature = torch.fx.Graph()
    
        # Replace all uses of tensor1 with tensor2 in the graph signature
        for node in graph_signature.nodes:
            if node.op == 'placeholder' and node.target == tensor1:
                node.target = tensor2
    
        return graph_signature
    