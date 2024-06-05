
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory.CUDAPluggableAllocator)
class CUDAPluggableAllocatorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_allocator_correctness(self):
        allocator = torch.cuda.memory.CUDAPluggableAllocator()
        result = allocator.allocator
        return result

    @test_api_version.larger_than("1.1.3")
    def test_allocator_large_scale(self):
        allocator = torch.cuda.memory.CUDAPluggableAllocator()
        result = allocator.allocator
        return result

    @test_api_version.larger_than("1.1.3")
    def test_type_correctness(self):
        allocator = torch.cuda.memory.CUDAPluggableAllocator()
        result = allocator.type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_type_large_scale(self):
        allocator = torch.cuda.memory.CUDAPluggableAllocator()
        result = allocator.type
        return result

@test_api(torch.cuda.memory)
class MemoryTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_caching_allocator_alloc_correctness(self):
        size = random.randint(1, 10)
        result = torch.cuda.memory.caching_allocator_alloc(size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_caching_allocator_alloc_large_scale(self):
        size = random.randint(1000, 10000)
        result = torch.cuda.memory.caching_allocator_alloc(size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_caching_allocator_delete_correctness(self):
        ptr = random.randint(1, 10)
        result = torch.cuda.memory.caching_allocator_delete(ptr)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_caching_allocator_delete_large_scale(self):
        ptr = random.randint(1000, 10000)
        result = torch.cuda.memory.caching_allocator_delete(ptr)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_change_current_allocator_correctness(self):
        allocator = torch.cuda.memory.CUDAPluggableAllocator()
        result = torch.cuda.memory.change_current_allocator(allocator)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_change_current_allocator_large_scale(self):
        allocator = torch.cuda.memory.CUDAPluggableAllocator()
        result = torch.cuda.memory.change_current_allocator(allocator)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_empty_cache_correctness(self):
        result = torch.cuda.memory.empty_cache()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_empty_cache_large_scale(self):
        result = torch.cuda.memory.empty_cache()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_allocator_backend_correctness(self):
        result = torch.cuda.memory.get_allocator_backend()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_allocator_backend_large_scale(self):
        result = torch.cuda.memory.get_allocator_backend()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_initialized_correctness(self):
        result = torch.cuda.memory.is_initialized()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_initialized_large_scale(self):
        result = torch.cuda.memory.is_initialized()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_list_gpu_processes_correctness(self):
        result = torch.cuda.memory.list_gpu_processes()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_list_gpu_processes_large_scale(self):
        result = torch.cuda.memory.list_gpu_processes()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_memory_allocated_correctness(self):
        result = torch.cuda.memory.max_memory_allocated()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_memory_allocated_large_scale(self):
        result = torch.cuda.memory.max_memory_allocated()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_memory_cached_correctness(self):
        result = torch.cuda.memory.max_memory_cached()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_memory_cached_large_scale(self):
        result = torch.cuda.memory.max_memory_cached()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_memory_reserved_correctness(self):
        result = torch.cuda.memory.max_memory_reserved()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_memory_reserved_large_scale(self):
        result = torch.cuda.memory.max_memory_reserved()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_mem_get_info_correctness(self):
        result = torch.cuda.memory.mem_get_info()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_mem_get_info_large_scale(self):
        result = torch.cuda.memory.mem_get_info()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_allocated_correctness(self):
        result = torch.cuda.memory.memory_allocated()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_allocated_large_scale(self):
        result = torch.cuda.memory.memory_allocated()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_cached_correctness(self):
        result = torch.cuda.memory.memory_cached()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_cached_large_scale(self):
        result = torch.cuda.memory.memory_cached()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_reserved_correctness(self):
        result = torch.cuda.memory.memory_reserved()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_reserved_large_scale(self):
        result = torch.cuda.memory.memory_reserved()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_snapshot_correctness(self):
        result = torch.cuda.memory.memory_snapshot()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_snapshot_large_scale(self):
        result = torch.cuda.memory.memory_snapshot()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_stats_correctness(self):
        result = torch.cuda.memory.memory_stats()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_stats_large_scale(self):
        result = torch.cuda.memory.memory_stats()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_stats_as_nested_dict_correctness(self):
        result = torch.cuda.memory.memory_stats_as_nested_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_stats_as_nested_dict_large_scale(self):
        result = torch.cuda.memory.memory_stats_as_nested_dict()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_summary_correctness(self):
        result = torch.cuda.memory.memory_summary()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_memory_summary_large_scale(self):
        result = torch.cuda.memory.memory_summary()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_accumulated_memory_stats_correctness(self):
        result = torch.cuda.memory.reset_accumulated_memory_stats()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_accumulated_memory_stats_large_scale(self):
        result = torch.cuda.memory.reset_accumulated_memory_stats()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_max_memory_allocated_correctness(self):
        result = torch.cuda.memory.reset_max_memory_allocated()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_max_memory_allocated_large_scale(self):
        result = torch.cuda.memory.reset_max_memory_allocated()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_max_memory_cached_correctness(self):
        result = torch.cuda.memory.reset_max_memory_cached()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_max_memory_cached_large_scale(self):
        result = torch.cuda.memory.reset_max_memory_cached()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_peak_memory_stats_correctness(self):
        result = torch.cuda.memory.reset_peak_memory_stats()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_peak_memory_stats_large_scale(self):
        result = torch.cuda.memory.reset_peak_memory_stats()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_per_process_memory_fraction_correctness(self):
        fraction = random.uniform(0.1, 1.0)
        result = torch.cuda.memory.set_per_process_memory_fraction(fraction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_per_process_memory_fraction_large_scale(self):
        fraction = random.uniform(0.1, 1.0)
        result = torch.cuda.memory.set_per_process_memory_fraction(fraction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_signature_correctness(self):
        result = torch.cuda.memory.signature()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_signature_large_scale(self):
        result = torch.cuda.memory.signature()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_segments_correctness(self):
        result = torch.cuda.memory.segments()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_segments_large_scale(self):
        result = torch.cuda.memory.segments()
        return result

