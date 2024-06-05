
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mps.Tensor)
class TorchMpsTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11")
    def test_tensor_abs(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.abs()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_abs_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.abs()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_neg(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.neg()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_neg_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.neg()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_positive(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.positive()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_positive_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.positive()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_pow(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        exponent = random.randint(1, 5)
        result = tensor.pow(exponent)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_pow_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        exponent = random.randint(1, 5)
        result = tensor.pow(exponent)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_abs(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.abs()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_abs_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.abs()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_absolute(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.absolute()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_absolute_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.absolute()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_acos(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.acos()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_acos_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.acos()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_acosh(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.acosh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_acosh_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.acosh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_add(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        alpha = random.uniform(0.1, 10.0)
        result = tensor1.add(tensor2, alpha=alpha)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_add_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        alpha = random.uniform(0.1, 10.0)
        result = tensor1.add(tensor2, alpha=alpha)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addbmm(self):
        batch_size = random.randint(1, 5)
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        k = random.randint(1, 10)
        tensor1 = torch.randn(batch_size, m, n)
        tensor2 = torch.randn(batch_size, k, n)
        tensor3 = torch.randn(batch_size, m, k)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addbmm(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addbmm_large_scale(self):
        batch_size = random.randint(100, 500)
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        k = random.randint(100, 1000)
        tensor1 = torch.randn(batch_size, m, n)
        tensor2 = torch.randn(batch_size, k, n)
        tensor3 = torch.randn(batch_size, m, k)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addbmm(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addcdiv(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        tensor3 = torch.randn(dim)
        value = random.uniform(0.1, 10.0)
        result = tensor1.addcdiv(tensor2, tensor3, value=value)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addcdiv_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        tensor3 = torch.randn(dim)
        value = random.uniform(0.1, 10.0)
        result = tensor1.addcdiv(tensor2, tensor3, value=value)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addcmul(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        tensor3 = torch.randn(dim)
        value = random.uniform(0.1, 10.0)
        result = tensor1.addcmul(tensor2, tensor3, value=value)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addcmul_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        tensor3 = torch.randn(dim)
        value = random.uniform(0.1, 10.0)
        result = tensor1.addcmul(tensor2, tensor3, value=value)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addmm(self):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        k = random.randint(1, 10)
        tensor1 = torch.randn(m, n)
        tensor2 = torch.randn(k, n)
        tensor3 = torch.randn(m, k)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addmm(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addmm_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        k = random.randint(100, 1000)
        tensor1 = torch.randn(m, n)
        tensor2 = torch.randn(k, n)
        tensor3 = torch.randn(m, k)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addmm(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addmv(self):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        tensor1 = torch.randn(m, n)
        tensor2 = torch.randn(n)
        tensor3 = torch.randn(m)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addmv(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addmv_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        tensor1 = torch.randn(m, n)
        tensor2 = torch.randn(n)
        tensor3 = torch.randn(m)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addmv(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addr(self):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        tensor1 = torch.randn(m, n)
        tensor2 = torch.randn(n)
        tensor3 = torch.randn(m)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addr(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_addr_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        tensor1 = torch.randn(m, n)
        tensor2 = torch.randn(n)
        tensor3 = torch.randn(m)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.addr(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_adjoint(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim, dim)
        result = tensor.adjoint()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_adjoint_large_scale(self):
        dim = random.randint(100, 1000)
        tensor = torch.randn(dim, dim)
        result = tensor.adjoint()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_align_as(self):
        dim1 = random.randint(1, 10)
        dim2 = random.randint(1, 10)
        tensor1 = torch.randn(dim1)
        tensor2 = torch.randn(dim2)
        result = tensor1.align_as(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_align_as_large_scale(self):
        dim1 = random.randint(100, 1000)
        dim2 = random.randint(100, 1000)
        tensor1 = torch.randn(dim1)
        tensor2 = torch.randn(dim2)
        result = tensor1.align_as(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_align_to(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        size = random.randint(1, 10)
        result = tensor.align_to(size)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_align_to_large_scale(self):
        dim = random.randint(100, 1000)
        tensor = torch.randn(dim)
        size = random.randint(100, 1000)
        result = tensor.align_to(size)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_all(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.all()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_all_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.all()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_allclose(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        result = tensor1.allclose(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_allclose_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        result = tensor1.allclose(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_amax(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.amax()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_amax_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.amax()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_amin(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.amin()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_amin_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.amin()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_aminmax(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.aminmax()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_aminmax_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.aminmax()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_angle(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim, dtype=torch.complex64)
        result = tensor.angle()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_angle_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim, dtype=torch.complex64)
        result = tensor.angle()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_any(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.any()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_any_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.any()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arccos(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.arccos()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arccos_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.arccos()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arccosh(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.arccosh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arccosh_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.arccosh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arcsin(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.arcsin()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arcsin_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.arcsin()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arcsinh(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.arcsinh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arcsinh_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.arcsinh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arctan(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.arctan()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arctan_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.arctan()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arctan2(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        result = tensor1.arctan2(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_arctan2_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randn(dim)
        tensor2 = torch.randn(dim)
        result = tensor1.arctan2(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_atanh(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.atanh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_atanh_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.atanh()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_backward(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim, requires_grad=True)
        result = tensor.backward()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_baddbmm(self):
        batch_size = random.randint(1, 5)
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        k = random.randint(1, 10)
        tensor1 = torch.randn(batch_size, m, n)
        tensor2 = torch.randn(batch_size, k, n)
        tensor3 = torch.randn(batch_size, m, k)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.baddbmm(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_baddbmm_large_scale(self):
        batch_size = random.randint(100, 500)
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        k = random.randint(100, 1000)
        tensor1 = torch.randn(batch_size, m, n)
        tensor2 = torch.randn(batch_size, k, n)
        tensor3 = torch.randn(batch_size, m, k)
        beta = random.uniform(0.1, 10.0)
        result = tensor1.baddbmm(tensor2, tensor3, beta=beta)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bernoulli(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.bernoulli()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bernoulli_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.bernoulli()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bfloat16(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = tensor.bfloat16()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bfloat16_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = tensor.bfloat16()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bincount(self):
        dim = random.randint(1, 10)
        tensor = torch.randint(0, 10, (dim,))
        result = tensor.bincount()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bincount_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randint(0, 1000, (dim,))
        result = tensor.bincount()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_and(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randint(0, 10, (dim,))
        tensor2 = torch.randint(0, 10, (dim,))
        result = tensor1.bitwise_and(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_and_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randint(0, 1000, (dim,))
        tensor2 = torch.randint(0, 1000, (dim,))
        result = tensor1.bitwise_and(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_left_shift(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randint(0, 10, (dim,))
        tensor2 = torch.randint(0, 10, (dim,))
        result = tensor1.bitwise_left_shift(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_left_shift_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randint(0, 1000, (dim,))
        tensor2 = torch.randint(0, 1000, (dim,))
        result = tensor1.bitwise_left_shift(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_not(self):
        dim = random.randint(1, 10)
        tensor = torch.randint(0, 10, (dim,))
        result = tensor.bitwise_not()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_not_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randint(0, 1000, (dim,))
        result = tensor.bitwise_not()
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_or(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randint(0, 10, (dim,))
        tensor2 = torch.randint(0, 10, (dim,))
        result = tensor1.bitwise_or(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_or_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randint(0, 1000, (dim,))
        tensor2 = torch.randint(0, 1000, (dim,))
        result = tensor1.bitwise_or(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_right_shift(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randint(0, 10, (dim,))
        tensor2 = torch.randint(0, 10, (dim,))
        result = tensor1.bitwise_right_shift(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_right_shift_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor1 = torch.randint(0, 1000, (dim,))
        tensor2 = torch.randint(0, 1000, (dim,))
        result = tensor1.bitwise_right_shift(tensor2)
        return result

    @test_api_version.larger_than("1.11")
    def test_tensor_bitwise_xor(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randint(0, 10, (dim,))
        tensor2 = torch.randint(0,