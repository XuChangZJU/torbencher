
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.ExecutionTraceObserver)
class TorchExecutionTraceObserverTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_ExecutionTraceObserver_type(self):
        result = torch.profiler.ExecutionTraceObserver.type
        return result

    @test_api_version.larger_than("1.12.0")
    def test_ExecutionTraceObserver_get_output_file_path(self):
        result = torch.profiler.ExecutionTraceObserver.get_output_file_path()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_ExecutionTraceObserver_is_running(self):
        result = torch.profiler.ExecutionTraceObserver.is_running()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_ExecutionTraceObserver_register_callback(self):
        callback = lambda: None
        result = torch.profiler.ExecutionTraceObserver.register_callback(callback)
        return result

    @test_api_version.larger_than("1.12.0")
    def test_ExecutionTraceObserver_start(self):
        result = torch.profiler.ExecutionTraceObserver.start()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_ExecutionTraceObserver_stop(self):
        result = torch.profiler.ExecutionTraceObserver.stop()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_ExecutionTraceObserver_unregister_callback(self):
        callback = lambda: None
        result = torch.profiler.ExecutionTraceObserver.unregister_callback(callback)
        return result

