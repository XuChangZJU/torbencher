import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


# 定义一个内存事件的模拟类
class MemoryEvent:
    def __init__(self, addr, size, is_alloc):
        self.addr = addr
        self.size = size
        self.is_alloc = is_alloc


# 假设的 MemRecordsAcc 类，用于演示
class MemRecordsAcc:
    def __init__(self, events):
        self.events = events
    
    def mem_records(self):
        return [event.__dict__ for event in self.events]


# 测试案例类
@test_api(torch.autograd.profiler_util.MemRecordsAcc)
class TorchAutogradProfilerutilMemrecordsaccTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mem_records_acc_correctness(self):
        # 生成一个内存事件
        address = random.randint(1, 1000)
        size = random.randint(1, 1000)
        alloc_dealloc = random.choice([True, False])
        
        # 创建 MemoryEvent 对象
        mem_event = MemoryEvent(address, size, alloc_dealloc)
        
        # 将单个事件放入列表
        mem_records = [mem_event]
        
        # 创建 MemRecordsAcc 对象
        mem_records_acc = MemRecordsAcc(mem_records)
        
        # 调用 mem_records 方法并获取结果
        result = mem_records_acc.mem_records()
        
        return result
    