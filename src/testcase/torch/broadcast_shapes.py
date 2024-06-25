import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.broadcast_shapes)
class TorchBroadcastshapesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_shapes_correctness(self):
        num_of_shapes = random.randint(2, 4)  # Random number of shapes to broadcast (minimum 2 for meaningful broadcasting)
        max_dim = random.randint(1, 4)  # Random maximum dimension size for the shapes
        
        shapes = []
        for _ in range(num_of_shapes):
            dim = random.randint(1, max_dim)  # Random dimension for each shape
            shape = []
            for _ in range(dim):
                shape.append(random.randint(1, 4))  # Random size for each dimension to ensure valid broadcasting
            shapes.append(torch.Size(shape))
        
        try:
            result_shape = torch.broadcast_shapes(*shapes)
            return result_shape
        except RuntimeError as e:
            return str(e)
    
    
    
    