import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch._assert)
class TorchAssertTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_assert_correctness(self):
        tensor_size = [random.randint(1, 4) for _ in range(random.randint(1, 4))]  # Random tensor dimensions with random number of elements
    
        condition_true = (torch.randn(tensor_size) > -1)  # Random tensor where the condition is always true (no element < -1)
        condition_false = (torch.randn(tensor_size) > 1)  # Random tensor where the condition is sometimes false (some elements <= 1)
    
        # Testing the condition that should always be true
        try:
            assert condition_true.all(), "Test message"
            assert_passed_true = True
        except AssertionError:
            assert_passed_true = False
    
        # Testing the condition that can be false
        try:
            assert condition_false.all(), "Test message"
            assert_passed_false = False
        except AssertionError:
            assert_passed_false = True
    
        return assert_passed_true, assert_passed_false
    
    
    
    