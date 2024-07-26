import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.cpp_extension.include_paths)
class TorchUtilsCppextensionIncludepathsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_include_paths_correctness(self):
        # Generate a random number of include paths
        num_paths = random.randint(1, 5)
        
        # Generate random include paths
        include_paths = [f"/path/to/include{random.randint(1, 100)}" for _ in range(num_paths)]
        
        # Call the function
        result = torch.utils.cpp_extension.include_paths()
        
        return result
    