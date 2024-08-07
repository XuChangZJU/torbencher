import torch
import random
from torch.export.graph_signature import InputSpec, InputKind, TensorArgument, SymIntArgument, ConstantArgument

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.graph_signature.InputSpec)
class TorchExportGraphUsignatureInputspecTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inputspec_tensor_argument(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Random tensor data
        tensor_data = torch.randn(input_size)
        tensor_arg = TensorArgument(tensor_data)

        # Random target string
        target = f"target_{random.randint(1, 100)}"

        # Create InputSpec with TensorArgument
        input_spec = InputSpec(InputKind.TENSOR, tensor_arg, target)
        return input_spec

    def test_inputspec_symint_argument(self):
        # Random integer value
        symint_value = random.randint(1, 100)
        symint_arg = SymIntArgument(symint_value)

        # Random target string
        target = f"target_{random.randint(1, 100)}"

        # Create InputSpec with SymIntArgument
        input_spec = InputSpec(InputKind.SYMBOLIC_INT, symint_arg, target)
        return input_spec

    def test_inputspec_constant_argument(self):
        # Random float value
        constant_value = random.uniform(0.1, 10.0)
        constant_arg = ConstantArgument(constant_value)

        # Random target string
        target = f"target_{random.randint(1, 100)}"

        # Create InputSpec with ConstantArgument
        input_spec = InputSpec(InputKind.CONSTANT, constant_arg, target)
        return input_spec
