
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.Event)
class TorchCpuEventTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_event_correctness(self):
        event = torch.cuda.Event()
        result = event.type()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_event_large_scale(self):
        event = torch.cuda.Event()
        result = event.type()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_event_query_correctness(self):
        event = torch.cuda.Event()
        event.record()
        result = event.query()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_event_query_large_scale(self):
        event = torch.cuda.Event()
        event.record()
        result = event.query()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_event_record_correctness(self):
        event = torch.cuda.Event()
        event.record()
        return None

    @test_api_version.larger_than("1.1.3")
    def test_event_record_large_scale(self):
        event = torch.cuda.Event()
        event.record()
        return None

    @test_api_version.larger_than("1.1.3")
    def test_event_synchronize_correctness(self):
        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        return None

    @test_api_version.larger_than("1.1.3")
    def test_event_synchronize_large_scale(self):
        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        return None

    @test_api_version.larger_than("1.1.3")
    def test_event_wait_correctness(self):
        event = torch.cuda.Event()
        event.record()
        event.wait()
        return None

    @test_api_version.larger_than("1.1.3")
    def test_event_wait_large_scale(self):
        event = torch.cuda.Event()
        event.record()
        event.wait()
        return None


