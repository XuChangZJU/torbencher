import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.expm1)
class TorchExpm1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_expm1_4d(self, input=None):
        if input is not None:
            result = torch.expm1(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.expm1(a)
        return [result, [a]]

@test_api(torch.special.i0)
class TorchSpecialI0TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_i0_4d(self, input=None):
        if input is not None:
            result = torch.special.i0(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.i0(a)
        return [result, [a]]

@test_api(torch.special.i1)
class TorchSpecialI1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_i1_4d(self, input=None):
        if input is not None:
            result = torch.special.i1(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.i1(a)
        return [result, [a]]

@test_api(torch.special.i0e)
class TorchSpecialI0eTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_i0e_4d(self, input=None):
        if input is not None:
            result = torch.special.i0e(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.i0e(a)
        return [result, [a]]

@test_api(torch.special.i1e)
class TorchSpecialI1eTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_i1e_4d(self, input=None):
        if input is not None:
            result = torch.special.i1e(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.i1e(a)
        return [result, [a]]

@test_api(torch.special.log_softmax)
class TorchSpecialLogSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_log_softmax_4d(self, input=None):
        if input is not None:
            result = torch.special.log_softmax(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.special.log_softmax(a, dim=1)
        return [result, [a, 1]]

@test_api(torch.special.softmax)
class TorchSpecialSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_softmax_4d(self, input=None):
        if input is not None:
            result = torch.special.softmax(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.special.softmax(a, dim=1)
        return [result, [a, 1]]

@test_api(torch.special.expit)
class TorchSpecialExpitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_expit_4d(self, input=None):
        if input is not None:
            result = torch.special.expit(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.expit(a)
        return [result, [a]]

@test_api(torch.special.logit)
class TorchSpecialLogitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_logit_4d(self, input=None):
        if input is not None:
            result = torch.special.logit(input[0])
            return [result, input]
        a = torch.rand(4).clamp(1e-6, 1 - 1e-6)  # Avoid values too close to 0 or 1
        result = torch.special.logit(a)
        return [result, [a]]

@test_api(torch.special.logsumexp)
class TorchSpecialLogsumexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_logsumexp_4d(self, input=None):
        if input is not None:
            result = torch.special.logsumexp(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.special.logsumexp(a, dim=1)
        return [result, [a, 1]]

@test_api(torch.special.digamma)
class TorchSpecialDigammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_digamma_4d(self, input=None):
        if input is not None:
            result = torch.special.digamma(input[0])
            return [result, input]
        a = torch.randn(4).abs()
        result = torch.special.digamma(a)
        return [result, [a]]

@test_api(torch.special.erf)
class TorchSpecialErfTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_erf_4d(self, input=None):
        if input is not None:
            result = torch.special.erf(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.erf(a)
        return [result, [a]]

@test_api(torch.special.erfc)
class TorchSpecialErfcTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_erfc_4d(self, input=None):
        if input is not None:
            result = torch.special.erfc(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.erfc(a)
        return [result, [a]]

@test_api(torch.special.erfcx)
class TorchSpecialErfcxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_erfcx_4d(self, input=None):
        if input is not None:
            result = torch.special.erfcx(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.erfcx(a)
        return [result, [a]]

@test_api(torch.special.ndtr)
class TorchSpecialNdtrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_ndtr_4d(self, input=None):
        if input is not None:
            result = torch.special.ndtr(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.ndtr(a)
        return [result, [a]]

@test_api(torch.special.ndtri)
class TorchSpecialNdtriTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_ndtri_4d(self, input=None):
        if input is not None:
            result = torch.special.ndtri(input[0])
            return [result, input]
        a = torch.rand(4).clamp(1e-6, 1 - 1e-6)
        result = torch.special.ndtri(a)
        return [result, [a]]

@test_api(torch.special.gammaln)
class TorchSpecialGammalnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_gammaln_4d(self, input=None):
        if input is not None:
            result = torch.special.gammaln(input[0])
            return [result, input]
        a = torch.randn(4).abs()
        result = torch.special.gammaln(a)
        return [result, [a]]

@test_api(torch.special.sinc)
class TorchSpecialSincTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_sinc_4d(self, input=None):
        if input is not None:
            result = torch.special.sinc(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.sinc(a)
        return [result, [a]]

@test_api(torch.special.log1p)
class TorchSpecialLog1pTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_log1p_4d(self, input=None):
        if input is not None:
            result = torch.special.log1p(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.log1p(a)
        return [result, [a]]

@test_api(torch.special.expm1)
class TorchSpecialExpm1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_expm1_4d(self, input=None):
        if input is not None:
            result = torch.special.expm1(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.expm1(a)
        return [result, [a]]

@test_api(torch.special.pow)
class TorchSpecialPowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_pow_4d(self, input=None):
        if input is not None:
            result = torch.special.pow(input[0], input[1])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.special.pow(a, b)
        return [result, [a, b]]

@test_api(torch.special.sigmoid)
class TorchSpecialSigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_sigmoid_4d(self, input=None):
        if input is not None:
            result = torch.special.sigmoid(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.special.sigmoid(a)
        return [result, [a]]
