import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.as_subclass)
class TorchTensorAsUsubclassTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_subclass_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)

        # Define a subclass of torch.Tensor
        class MyTensor(torch.Tensor):
            pass

        # Ensure MyTensor is a subclass of torch.Tensor
        assert issubclass(MyTensor, torch.Tensor)

        # Use as_subclass to create an instance of MyTensor with the same data as tensor
        subclass_tensor = tensor.as_subclass(MyTensor)

        # Check if the subclass_tensor is an instance of MyTensor
        assert isinstance(subclass_tensor, MyTensor)

        # Check if changes in subclass_tensor reflect in the original tensor
        subclass_tensor += 1
        assert torch.equal(tensor, subclass_tensor)

        return subclass_tensor
