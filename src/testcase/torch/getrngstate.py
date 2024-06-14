import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.getrngstate)
class TorchGetrngstateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_rng_state_correctness(self):
    # Generate random tensors before getting the RNG state
    dim = random.randint(1, 4)  
    num_of_elements_each_dim = random.randint(1,5) 
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    tensor1 = torch.randn(input_size)
    tensor2 = torch.randn(input_size)

    # Get the current RNG state
    rng_state = torch.get_rng_state() 

    # Perform some operations that use the RNG
    result1 = torch.randn(input_size) 

    # Reset the RNG state
    torch.set_rng_state(rng_state)

    # Perform the same operations again 
    result2 = torch.randn(input_size)

    # result1 and result2 should be the same because we reset the RNG state
    return result1, result2
