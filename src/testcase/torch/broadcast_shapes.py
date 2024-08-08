import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.broadcast_shapes)
class TorchBroadcastUshapesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_shapes_correctness(self):
        num_of_shapes = random.randint(2,
                                       4)  # Random number of shapes to broadcast (minimum 2 for meaningful broadcasting)
        max_dim = random.randint(1, 4)  # Random maximum dimension size for the shapes

        shapes = []
        for idx in range(num_of_shapes):
            dim = random.randint(1, max_dim)  # Random dimension for each shape
            shape = []
            for i in range(dim):
                if idx == 0:
                    shape.append(random.randint(1, 4))  # Random size for each dimension to ensure valid broadcasting
                else:
                    shape.append(
                        random.randint(1, 3) * shapes[0][i % len(shapes[0])])  # Correct indexing for dimensions
            shapes.append(torch.Size(shape))

        result_shape = torch.broadcast_shapes(*shapes)


        return result_shape
