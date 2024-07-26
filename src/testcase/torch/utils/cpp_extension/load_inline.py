import torch
import random
import subprocess
from torch.utils.cpp_extension import load_inline

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.cpp_extension.load_inline)
class TorchUtilsCppextensionLoadinlineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_load_inline_correctness(self):
        # Ensure Ninja is installed
        try:
            subprocess.run(["ninja", "--version"], check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("Ninja is required to load C++ extensions. Please install it using 'pip install ninja'.")
    
        # Randomly generate a name for the extension
        extension_name = f"extension_{random.randint(1, 1000)}"
        
        # Randomly generate a simple C++ source code
        cpp_source = """
        #include <torch/extension.h>
        torch::Tensor add(torch::Tensor a, torch::Tensor b) {
            return a + b;
        }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("add", &add, "A function that adds two tensors");
        }
        """
        
        # Randomly generate a simple CUDA source code
        cuda_source = """
        __global__ void add_kernel(float* a, float* b, float* c, int size) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size) {
                c[index] = a[index] + b[index];
            }
        }
        
        torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
            auto c = torch::zeros_like(a);
            int size = a.numel();
            int threads = 1024;
            int blocks = (size + threads - 1) / threads;
            add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);
            return c;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("add_cuda", &add_cuda, "A function that adds two tensors using CUDA");
        }
        """
        
        # Load the inline extension
        extension = load_inline(extension_name, cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-std=c++14'])
        
        # Randomly generate tensor sizes
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Generate random tensors
        tensor1 = torch.randn(input_size, device='cuda')
        tensor2 = torch.randn(input_size, device='cuda')
        
        # Test the C++ function
        result_cpp = extension.add(tensor1.cpu(), tensor2.cpu())
        
        # Test the CUDA function
        result_cuda = extension.add_cuda(tensor1, tensor2)
        
        return result_cpp, result_cuda
    