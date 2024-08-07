import torch
import random
from torch.export import graph_signature

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.graph_signature.OutputSpec)
class TorchExportGraphUsignatureOutputspecTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_OutputSpec_correctness(self):
        # Generate random parameters for OutputSpec
        output_kind = random.choice(list(graph_signature.OutputKind))  # Randomly choose an OutputKind
        arg_type = random.randint(0,
                                  2)  # Randomly choose an argument type: 0 for TensorArgument, 1 for SymIntArgument, 2 for ConstantArgument

        # Generate argument based on arg_type
        if arg_type == 0:  # TensorArgument
            dim = random.randint(1, 4)
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for i in range(dim)]
            arg = graph_signature.TensorArgument(torch.randn(input_size))
        elif arg_type == 1:  # SymIntArgument
            arg = graph_signature.SymIntArgument(random.randint(1, 10))
        else:  # ConstantArgument
            arg = graph_signature.ConstantArgument(torch.tensor(random.uniform(0.1, 10.0)))

        # Generate random target string
        target = "output_" + str(random.randint(0, 10))

        # Create OutputSpec object
        output_spec = graph_signature.OutputSpec(output_kind, arg, target)

        return output_spec
