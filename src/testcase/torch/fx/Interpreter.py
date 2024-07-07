import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fx.Interpreter)
class TorchFxInterpreterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_interpreter_correctness(self):
        class NegSigmSwapInterpreter(torch.fx.Interpreter):
            def call_function(self, target, args, kwargs):
                if target == torch.sigmoid:
                    return torch.neg(*args, **kwargs)
                return super().call_function(target, args, kwargs)
    
            def call_method(self, target, args, kwargs):
                if target == 'neg':
                    call_self, *args_tail = args
                    return call_self.sigmoid(*args_tail, **kwargs)
                return super().call_method(target, args, kwargs)
    
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        input_tensor = torch.randn(input_size)
    
        def fn(x):
            return torch.sigmoid(x).neg()
    
        gm = torch.fx.symbolic_trace(fn)
        interpreter = NegSigmSwapInterpreter(gm)
        result = interpreter.run(input_tensor)
        return result
    
    
    
    