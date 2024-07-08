import torch
import random
import tempfile

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.autograd.profiler.profile.export_chrome_trace)
class TorchAutogradProfilerProfileExportchrometraceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_chrome_trace_correctness(self):
        # Generate random parameters
        path = tempfile.mkdtemp() + "/trace.json" # Generate a temporary file path
    
        # Generate random input for profile
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            dim = random.randint(1, 4)  # Random dimension for the tensors
            num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
            input_size=[num_of_elements_each_dim for i in range(dim)] 
            tensor1 = torch.randn(input_size)
            tensor2 = torch.randn(input_size)
            result = torch.add(tensor1, tensor2)
    
        # Call the function to be tested
        result = prof.export_chrome_trace(path)
    
        return result
    