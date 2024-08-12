import random

import torch
from torch.export import ModuleCallSignature
from torch.export.graph_signature import TensorArgument, SymIntArgument, ConstantArgument
from torch.utils._pytree import TreeSpec

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.ModuleCallSignature)
class TorchExportModulecallsignatureTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_module_call_signature_correctness(self):
        # Randomly generate the number of inputs and outputs
        num_inputs = random.randint(1, 4)
        num_outputs = random.randint(1, 4)

        # Generate random TensorArguments for inputs and outputs
        inputs = [TensorArgument("tensor") for _ in range(num_inputs)]
        outputs = [TensorArgument("tensor") for _ in range(num_outputs)]

        # Generate random SymIntArguments for inputs and outputs
        inputs += [SymIntArgument("symint") for _ in range(random.randint(0, 2))]
        outputs += [SymIntArgument("symint") for _ in range(random.randint(0, 2))]

        # Generate random ConstantArguments for inputs and outputs
        inputs += [ConstantArgument(random.uniform(0.1, 10.0)) for _ in range(random.randint(0, 2))]
        outputs += [ConstantArgument(random.uniform(0.1, 10.0)) for _ in range(random.randint(0, 2))]

        # Generate random TreeSpec for in_spec and out_spec
        in_spec = TreeSpec(None, None, [])
        out_spec = TreeSpec(None, None, [])

        # Create ModuleCallSignature instance
        module_call_signature = ModuleCallSignature(inputs, outputs, in_spec, out_spec)

        return module_call_signature
