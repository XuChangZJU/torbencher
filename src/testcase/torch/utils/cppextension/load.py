import torch
import random
from torch.utils.cpp_extension import load
import os


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.cpp_extension.load)
class TorchUtilsCppextensionLoadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cpp_extension_load_correctness(self):
        # Randomly generate the name of the extension
        extension_name = f"extension_{random.randint(1, 1000)}"
        
        # Randomly generate the source file name
        source_file_name = f"source_{random.randint(1, 1000)}.cpp"
        
        # Create a dummy source file with minimal valid C++ code for PyTorch extension
        with open(source_file_name, 'w') as f:
            f.write("""
            #include <torch/extension.h>
            torch::Tensor add(torch::Tensor a, torch::Tensor b) {
                return a + b;
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("add", &add, "A function that adds two tensors");
            }
            """)
        
        # Load the extension
        extension = load(name=extension_name, sources=[source_file_name], verbose=True)
        
        # Generate random tensors for testing the loaded extension
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        
        # Use the loaded extension to add the tensors
        result = extension.add(tensor1, tensor2)
        
        # Clean up the generated source file
        os.remove(source_file_name)
        
        return result
    
    
    
    