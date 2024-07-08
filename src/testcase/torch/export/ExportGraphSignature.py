import torch
import random
from torch.fx import symbolic_trace
from torch.export import ExportGraphSignature, export

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.export.ExportGraphSignature)
class TorchExportExportgraphsignatureTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_graph_signature_correctness(self):
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super(CustomModule, self).
                self.my_parameter = torch.nn.Parameter(torch.tensor(random.uniform(1.0, 10.0)))
                self.register_buffer('my_buffer1', torch.tensor(random.uniform(1.0, 10.0)))
                self.register_buffer('my_buffer2', torch.tensor(random.uniform(1.0, 10.0)))
    
            def forward(self, x1, x2):
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2
                self.my_buffer2.add_(random.uniform(0.1, 1.0))
                return output
    
        # Create random input tensors
        input_size = [random.randint(1, 4) for _ in range(2)]
        x1 = torch.randn(input_size)
        x2 = torch.randn(input_size)
    
        # Instantiate the module and trace it
        module = CustomModule()
        traced_graph = symbolic_trace(module)
    
        # Export the graph
        exported_program = export(traced_graph, (x1, x2))
    
        # Check the ExportGraphSignature
        assert isinstance(exported_program.graph_signature, ExportGraphSignature), "Signature is not of type ExportGraphSignature"
        assert len(exported_program.graph_signature.input_specs) == 5, "Incorrect number of input specs"
        assert len(exported_program.graph_signature.output_specs) == 2, "Incorrect number of output specs"
    
        return exported_program.graph_signature
    