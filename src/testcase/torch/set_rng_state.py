import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.set_rng_state)
class TorchSetrngstateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_rng_state_correctness(self):
        # Generating a random seed
        seed = random.randint(0, 10000)

        # Setting the seed to ensure reproducibility
        torch.manual_seed(seed)

        # Creating a random ByteTensor to represent the desired state
        # The size of the state tensor should match the expected size for the current PyTorch version
        new_state = torch.get_rng_state()

        # Setting the RNG state with the created ByteTensor
        torch.set_rng_state(new_state)

        # Generating a random tensor to verify RNG state
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_elements_per_dim = random.randint(1, 5)  # Random number of elements per dimension
        input_size = [num_elements_per_dim for _ in range(dim)]

        random_tensor = torch.randn(input_size)  # Generating a random tensor

        return random_tensor
