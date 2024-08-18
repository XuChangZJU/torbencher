import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Generator)
class TorchGeneratorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_generator_correctness(self):
        # Create generator (Note: torch.Generator() does not support device argument)
        generator = torch.Generator()

        # Use the generator to create a seeded random tensor
        seed = random.randint(1, 1000)
        generator.manual_seed(seed)

        # Capture the state of the generator after setting the seed
        state_after_setting_seed = generator.get_state()

        # Reset the generator to the same seed
        generator.manual_seed(seed)

        # Capture the state of the generator after resetting the seed
        state_after_resetting_seed = generator.get_state()

        # Verify if the states of the generator from the same seed are equal
        return state_after_setting_seed == state_after_resetting_seed
