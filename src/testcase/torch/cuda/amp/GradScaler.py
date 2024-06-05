
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.amp.GradScaler)
class GradScalerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_type(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.type
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_type_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.type
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_backoff_factor(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_backoff_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_backoff_factor_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_backoff_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_growth_factor(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_growth_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_growth_factor_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_growth_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_growth_interval(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_growth_interval()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_growth_interval_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_growth_interval()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_scale()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_get_scale_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.get_scale()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_is_enabled(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.is_enabled()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_is_enabled_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.is_enabled()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_load_state_dict(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        state_dict = grad_scaler.state_dict()
        grad_scaler.load_state_dict(state_dict)
        result = grad_scaler.state_dict()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_load_state_dict_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        state_dict = grad_scaler.state_dict()
        grad_scaler.load_state_dict(state_dict)
        result = grad_scaler.state_dict()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        tensor = torch.randn(10)
        result = grad_scaler.scale(tensor)
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_scale_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        tensor = torch.randn(10000)
        result = grad_scaler.scale(tensor)
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_set_backoff_factor(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        backoff_factor = random.uniform(0.1, 10.0)
        grad_scaler.set_backoff_factor(backoff_factor)
        result = grad_scaler.get_backoff_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_set_backoff_factor_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        backoff_factor = random.uniform(0.1, 10.0)
        grad_scaler.set_backoff_factor(backoff_factor)
        result = grad_scaler.get_backoff_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_set_growth_factor(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        growth_factor = random.uniform(0.1, 10.0)
        grad_scaler.set_growth_factor(growth_factor)
        result = grad_scaler.get_growth_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_set_growth_factor_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        growth_factor = random.uniform(0.1, 10.0)
        grad_scaler.set_growth_factor(growth_factor)
        result = grad_scaler.get_growth_factor()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_set_growth_interval(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        growth_interval = random.randint(1, 10)
        grad_scaler.set_growth_interval(growth_interval)
        result = grad_scaler.get_growth_interval()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_set_growth_interval_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        growth_interval = random.randint(1, 10)
        grad_scaler.set_growth_interval(growth_interval)
        result = grad_scaler.get_growth_interval()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_state_dict(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.state_dict()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_state_dict_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        result = grad_scaler.state_dict()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_step(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.SGD(torch.randn(10, requires_grad=True), lr=0.01)
        loss = torch.randn(10)
        optimizer.zero_grad()
        loss.backward()
        grad_scaler.step(optimizer)
        result = optimizer.param_groups[0]['params'][0].grad
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_step_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.SGD(torch.randn(10000, requires_grad=True), lr=0.01)
        loss = torch.randn(10000)
        optimizer.zero_grad()
        loss.backward()
        grad_scaler.step(optimizer)
        result = optimizer.param_groups[0]['params'][0].grad
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_update(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.SGD(torch.randn(10, requires_grad=True), lr=0.01)
        loss = torch.randn(10)
        optimizer.zero_grad()
        loss.backward()
        grad_scaler.update()
        result = grad_scaler.get_scale()
        return result

    @test_api_version.larger_than("1.6.0")
    def test_grad_scaler_update_large_scale(self):
        grad_scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.SGD(torch.randn(10000, requires_grad=True), lr=0.01)
        loss = torch.randn(10000)
        optimizer.zero_grad()
        loss.backward()
        grad_scaler.update()
        result = grad_scaler.get_scale()
        return result

