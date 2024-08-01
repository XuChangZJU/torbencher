import torch
import random
from torch.utils.checkpoint import checkpoint_sequential

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.checkpoint.checkpoint_sequential)
class TorchUtilsCheckpointCheckpointsequentialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_checkpoint_sequential_correctness(self):
        # Randomly generate the number of layers
        num_layers = random.randint(2, 5)

        # Create a list of random linear layers
        layer_sizes = [random.randint(1, 10) for _ in range(num_layers + 1)]
        layers = [torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(num_layers)]

        # Randomly generate the number of segments
        num_segments = random.randint(1, num_layers)

        # Random input tensor
        input_tensor = torch.randn(random.randint(1, 10), layer_sizes[0])

        # Apply checkpoint_sequential
        result = checkpoint_sequential(layers, num_segments, input_tensor)
        return result
