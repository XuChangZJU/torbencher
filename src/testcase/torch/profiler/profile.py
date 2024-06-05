
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.profile)
class TorchProfileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_profile_type(self):
        result = torch.profiler.profile.type
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_add_metadata(self):
        metadata_key = "test_key"
        metadata_value = "test_value"
        result = torch.profiler.profile.add_metadata(metadata_key, metadata_value)
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_add_metadata_json(self):
        metadata_json = '{"test_key": "test_value"}'
        result = torch.profiler.profile.add_metadata_json(metadata_json)
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_events(self):
        result = torch.profiler.profile.events
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_export_chrome_trace(self):
        output_file_path = "test_output.json"
        result = torch.profiler.profile.export_chrome_trace(output_file_path)
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_export_memory_timeline(self):
        output_file_path = "test_output.json"
        result = torch.profiler.profile.export_memory_timeline(output_file_path)
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_export_stacks(self):
        output_file_path = "test_output.json"
        result = torch.profiler.profile.export_stacks(output_file_path)
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_key_averages(self):
        result = torch.profiler.profile.key_averages
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_prepare_trace(self):
        result = torch.profiler.profile.prepare_trace()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_start(self):
        result = torch.profiler.profile.start()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_start_trace(self):
        result = torch.profiler.profile.start_trace()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_step(self):
        result = torch.profiler.profile.step()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_stop(self):
        result = torch.profiler.profile.stop()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_profile_stop_trace(self):
        result = torch.profiler.profile.stop_trace()
        return result

