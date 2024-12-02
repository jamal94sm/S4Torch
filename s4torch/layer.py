

"""

    S4 Layer

"""

from typing import Union

import numpy as np
import torch
from torch import nn
from torch import view_as_real as as_real
from torch.fft import ifft, irfft, rfft
from torch.nn import functional as F
from torch.nn import init


def _log_step_initializer(
    tensor: torch.Tensor,  # values should be from U(0, 1)
    dt_min: float = 0.001,
    dt_max: float = 0.1,
) -> torch.Tensor:
    scale = np.log(dt_max) - np.log(dt_min)
    return tensor * scale + np.log(dt_min)


def _make_omega_l(l_max: int, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    return torch.arange(l_max).type(dtype).mul(2j * np.pi / l_max).exp()


def _make_hippo(N: int) -> np.ndarray:
    def idx2value(n: int, k: int) -> Union[int, float]:
        if n > k:
            return np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    hippo = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hippo[i, j] = idx2value(i + 1, j + 1)
    return hippo

def _make_diagonal(N: int) -> np.ndarray:
    def idx2value(n: int, k: int) -> Union[int, float]:
        if n == k:
            return n + 1
        else:
            return 0

    hippo = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hippo[i, j] = idx2value(i + 1, j + 1)
    return hippo



def diag_matrix_pow (A, l):
      x = np.diag(A)
      y = x**l
      return np.diag(y)



def _make_nplr_hippo(N: int) -> tuple[np.ndarray, ...]:
    nhippo = -1 * _make_diagonal(N)

    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]

    lambda_, V = np.linalg.eig(S)
    return lambda_, p, q, V



def _make_p_q_lambda(n: int) -> list[torch.Tensor]:
    lambda_, p, q, V = _make_nplr_hippo(n)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    return [torch.from_numpy(i) for i in (p, q, lambda_)]



def _cauchy_dot(v: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if v.ndim == 1:
        v = v.unsqueeze(0).unsqueeze(0)
    elif v.ndim == 2:
        v = v.unsqueeze(1)
    elif v.ndim != 3:
        raise IndexError(f"Expected `v` to be 1D, 2D or 3D, got {v.ndim}D")
    return (v / denominator).sum(dim=-1)



def _non_circular_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    l_max = u.shape[1]
    ud = rfft(F.pad(u.float(), pad=(0, 0, 0, l_max, 0, 0)), dim=1)
    Kd = rfft(F.pad(K.float(), pad=(0, l_max)), dim=-1)
    return irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1).type_as(u)



class S4Layer(nn.Module):
    def __init__(self, d_model: int, n: int, l_max: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max

        self.B = nn.Parameter(init.xavier_normal_(torch.empty(n, d_model))) 
        self.A = torch.nn.Parameter(torch.tensor(_make_diagonal(n), requires_grad=True).type_as(self.B))
        self.C = nn.Parameter(init.xavier_normal_(torch.empty(d_model, n)))
        self.D = nn.Parameter(torch.ones(1, 1, d_model))
        self.step = nn.Parameter(_log_step_initializer(torch.rand(d_model))).exp()

    
    
    @property
    def K(self) -> torch.Tensor:  # noqa
      a = torch.tensor(self.A, requires_grad=False)
      a = np.array(a)
      b = torch.tensor(self.B, requires_grad=False)
      b = np.array(b)
      c = torch.tensor(self.C, requires_grad=False)
      c = np.array(c)
      s = torch.tensor(self.step, requires_grad=False)
      s = np.array(s)
    
      I = np.eye(a.shape[0])
      Ab = a
      Bb = b
      Cb = c
      K = []
      for i in range(b.shape[1]):
        BL = inv(I - (s[i] / 2.0) * a)
        Ab = BL @ (I + (s[i] / 2.0) * a)
        Bb[:,i] = (BL * s[i]) @ (b[:,i])
        #k = np.array([(Cb[i,:] @ matrix_power(Ab, l) @ Bb[:,i]).reshape() for l in range(self.l_max)])
        k = np.array([(Cb[i,:] @ diag_matrix_pow(Ab, l) @ Bb[:,i]) for l in range(self.l_max)])
        K.append(k)
    
      K = np.array(K)
      K = torch.tensor(K, requires_grad=True).unsqueeze(0)
    
      return K



    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
        return _non_circular_convolution(u, K=self.K) + (self.D * u)

if __name__ == "__main__":
    N = 32
    d_model = 128
    l_max = 784

    u = torch.randn(1, l_max, d_model)

    s4layer = S4Layer(d_model, n=N, l_max=l_max)
    assert s4layer(u).shape == u.shape
