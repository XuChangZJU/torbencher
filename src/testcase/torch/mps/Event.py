
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mps.Event)
class TorchMpsEventTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11")
    def test_event_type(self):
        event = torch.mps.Event()
        result = event.type()
        return result

    @test_api_version.larger_than("1.11")
    def test_event_elapsed_time(self):
        event = torch.mps.Event()
        event.record()
        torch.mps.synchronize()
        result = event.elapsed_time()
        return result

    @test_api_version.larger_than("1.11")
    def test_event_query(self):
        event = torch.mps.Event()
        event.record()
        result = event.query()
        return result

    @test_api_version.larger_than("1.11")
    def test_event_record(self):
        event = torch.mps.Event()
        result = event.record()
        return result

    @test_api_version.larger_than("1.11")
    def test_event_synchronize(self):
        event = torch.mps.Event()
        result = event.synchronize()
        return result

    @test_api_version.larger_than("1.11")
    def test_event_wait(self):
        event = torch.mps.Event()
        event.record()
        result = event.wait()
        return result

