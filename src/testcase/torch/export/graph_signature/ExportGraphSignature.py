import torch
import random
from torch.export.graph_signature import ExportGraphSignature, InputSpec, OutputSpec, InputKind, OutputKind, \
    TensorArgument

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.graph_signature.ExportGraphSignature)
class TorchExportGraphsignatureExportgraphsignatureTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_graph_signature_correctness(self):
        # Randomly generate tensor sizes
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Randomly generate tensors for parameters and buffers
        my_parameter = torch.randn(input_size)
        my_buffer1 = torch.randn(input_size)
        my_buffer2 = torch.randn(input_size)

        # Randomly generate user inputs
        user_input1 = torch.randn(input_size)
        user_input2 = torch.randn(input_size)

        # Define the ExportGraphSignature
        export_graph_signature = ExportGraphSignature(
            input_specs=[
                InputSpec(kind=InputKind.PARAMETER, arg=TensorArgument(name='arg0_1'), target='my_parameter'),
                InputSpec(kind=InputKind.BUFFER, arg=TensorArgument(name='arg1_1'), target='my_buffer1'),
                InputSpec(kind=InputKind.BUFFER, arg=TensorArgument(name='arg2_1'), target='my_buffer2'),
                InputSpec(kind=InputKind.USER_INPUT, arg=TensorArgument(name='arg3_1'), target=None),
                InputSpec(kind=InputKind.USER_INPUT, arg=TensorArgument(name='arg4_1'), target=None)
            ],
            output_specs=[
                OutputSpec(kind=OutputKind.BUFFER_MUTATION, arg=TensorArgument(name='add_2'), target='my_buffer2'),
                OutputSpec(kind=OutputKind.USER_OUTPUT, arg=TensorArgument(name='add_1'), target=None)
            ]
        )

        return export_graph_signature
