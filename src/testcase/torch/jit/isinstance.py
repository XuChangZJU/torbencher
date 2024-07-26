import torch
import random
from typing import List, Dict, Tuple, Optional, Any

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.isinstance)
class TorchJitIsinstanceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isinstance_correctness(self):
        # Randomly choose a type to test
        type_choice = random.choice(
            ['List[torch.Tensor]', 'Dict[str, str]', 'Optional[Tuple[int, str, int]]', 'bool', 'int'])

        if type_choice == 'List[torch.Tensor]':
            # Generate a random list of tensors
            dim = random.randint(1, 4)
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for _ in range(dim)]
            input_data = [torch.randn(input_size) for _ in range(random.randint(1, 5))]
            target_type = List[torch.Tensor]

        elif type_choice == 'Dict[str, str]':
            # Generate a random dictionary with string keys and values
            input_data = {f"key{random.randint(1, 10)}": f"val{random.randint(1, 10)}" for _ in
                          range(random.randint(1, 5))}
            target_type = Dict[str, str]

        elif type_choice == 'Optional[Tuple[int, str, int]]':
            # Generate a random tuple with int, str, int or None
            if random.choice([True, False]):
                input_data = (random.randint(1, 10), f"str{random.randint(1, 10)}", random.randint(1, 10))
            else:
                input_data = None
            target_type = Optional[Tuple[int, str, int]]

        elif type_choice == 'bool':
            # Generate a random boolean value
            input_data = random.choice([True, False])
            target_type = bool

        elif type_choice == 'int':
            # Generate a random integer
            input_data = random.randint(1, 100)
            target_type = int

        result = torch.jit.isinstance(input_data, target_type)
        return result
