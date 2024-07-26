import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.export.ExportedProgram)
class TorchExportExportedprogramTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exported_program_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random tensor data
        tensor_data = torch.randn(input_size)
    
        # Define a simple model to export
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x * 2
    
        model = SimpleModel()
    
        # Export the model
        scripted_model = torch.jit.script(model)
    
        # Perform a simple transformation: multiply all outputs by 3
        for node in scripted_model.graph.nodes():
            if node.kind() == 'prim::Return':
                with scripted_model.graph.inserting_before(node):
                    new_node = scripted_model.graph.create('aten::mul', (node.inputsAt(0), torch.tensor(3)))
                    scripted_model.graph.insertNode(new_node)
                    node.replaceInput(0, new_node.output())
    
        # Recompile the graph module
        scripted_model.graph.lint()
    
        # Call the transformed exported program
        result = scripted_model(tensor_data)
        return result
    