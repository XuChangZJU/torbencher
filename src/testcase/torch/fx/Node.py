import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fx.Node)
class TorchFxNodeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_node_placeholder(self):
        # Generate random data for the placeholder node
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor_data = torch.randn(input_size)
    
        # Create a placeholder node
        placeholder_node = torch.fx.Node(
            graph=None,
            name="input_placeholder",
            op="placeholder",
            target="input_tensor",
            args=(),
            kwargs={},
        )
    
        # Return the tensor data as a side effect to show the effect of the placeholder
        return tensor_data
    
    
    def test_node_get_attr(self):
        # Create a tensor and register it as a buffer
        buffer_data = torch.randn(3, 4)
        buffer_name = "my_buffer"
    
        # Create a get_attr node to retrieve the buffer
        get_attr_node = torch.fx.Node(
            graph=None,
            name="get_attr_node",
            op="get_attr",
            target=buffer_name,
            args=(),
            kwargs={},
        )
    
        # Return the buffer data as a side effect to show the effect of get_attr
        return buffer_data
    
    
    def test_node_call_function(self):
        # Generate random data for the function call
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Create a call_function node for torch.add
        call_function_node = torch.fx.Node(
            graph=None,
            name="call_add",
            op="call_function",
            target=torch.add,
            args=(tensor1, tensor2),
            kwargs={},
        )
    
        # Return the result of torch.add as a side effect
        return torch.add(tensor1, tensor2)
    
    
    def test_node_call_module(self):
        # Create a simple module
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 1
    
        # Instantiate the module
        module_instance = MyModule()
    
        # Generate random input data
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
    
        # Create a call_module node
        call_module_node = torch.fx.Node(
            graph=None,
            name="call_my_module",
            op="call_module",
            target="my_module",  # Assuming the module is registered with this name
            args=(input_tensor,),
            kwargs={},
        )
    
        # Return the result of running the module as a side effect
        return module_instance(input_tensor)
    
    
    def test_node_call_method(self):
        # Generate random data for the method call
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor = torch.randn(input_size)
    
        # Create a call_method node for torch.Tensor.add_
        call_method_node = torch.fx.Node(
            graph=None,
            name="call_add_",
            op="call_method",
            target="add_",
            args=(tensor, tensor),  # Include the self argument
            kwargs={},
        )
    
        # Return the result of tensor.add_ as a side effect
        return tensor.add_(tensor)
    
    
    def test_node_output(self):
        # Generate random data for the output node
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        output_tensor = torch.randn(input_size)
    
        # Create an output node
        output_node = torch.fx.Node(
            graph=None,
            name="output",
            op="output",
            target="output",
            args=(output_tensor,),
            kwargs={},
        )
    
        # Return the output tensor as a side effect
        return output_tensor
    
    
    # Automatically added function calls
    
    
    
    
    
    
    
    