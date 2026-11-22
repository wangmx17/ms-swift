"""Case-local GDN handoff recomputation using the unmodified upstream kernels."""
import importlib
import torch

_backend = importlib.import_module('torch_musa_op_extension.attention.tilelang.flash_linear_attention.gdn_chunk_bwd')

class _Context:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

class RecomputeGDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state, scale, output_final_state,
                cu_seqlens=None, lengths=None):
        scratch = _Context()
        output = _backend._GatedDeltaNet.forward(
            scratch, q, k, v, g, beta, initial_state, scale,
            output_final_state, cu_seqlens, lengths)
        ctx.save_for_backward(q, k, v, g, beta, initial_state, cu_seqlens)
        ctx.options = scale, output_final_state, lengths
        return output

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, beta, initial_state, cu_seqlens = ctx.saved_tensors
        scale, output_final_state, lengths = ctx.options
        scratch = _Context()
        with torch.no_grad():
            output = _backend._GatedDeltaNet.forward(
                scratch, q, k, v, g, beta, initial_state, scale,
                output_final_state, cu_seqlens, lengths)
            del output
            return _backend._GatedDeltaNet.backward(scratch, do, dht)


def gdn_chunk(q, k, v, g, beta, *, scale=None, initial_state=None,
              output_final_state=True, cu_seqlens=None):
    _backend._check_inputs(q, k, v, g, beta, scale)
    lengths = None
    if cu_seqlens is not None:
        lengths = _backend._check_varlen(q, v, cu_seqlens, initial_state, output_final_state)
    q, k, v, g, beta = (_backend._tl_contiguous(x) for x in (q, k, v, g, beta))
    return RecomputeGDN.apply(q, k, v, g, beta, initial_state, scale,
                             output_final_state, cu_seqlens, lengths)
