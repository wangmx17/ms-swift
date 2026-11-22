"""Joint frozen output-head/CE recomputation, preserving actual upstream gradients."""
import torch

class ChunkedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, labels, chunk, loss_call):
        assert not weight.requires_grad and x.ndim == 3 and x.shape[1] == 1
        assert labels.shape == x.shape[:2] and chunk > 0
        ctx.save_for_backward(x, weight, labels)
        ctx.chunk = chunk
        ctx.loss_call = loss_call
        # Function.forward has autograd disabled; logits and CE's dlogits die
        # after each chunk. Retain only hidden states, frozen weights and labels.
        return torch.cat([loss_call(x[i:i+chunk], weight, labels[i:i+chunk])
                          for i in range(0, x.shape[0], chunk)], dim=0)

    @staticmethod
    def backward(ctx, upstream):
        x, weight, labels = ctx.saved_tensors
        gradient = torch.empty_like(x)
        for begin in range(0, x.shape[0], ctx.chunk):
            end = min(begin+ctx.chunk, x.shape[0])
            with torch.enable_grad():
                part = x[begin:end].detach().requires_grad_()
                loss = ctx.loss_call(part, weight, labels[begin:end])
                dx = torch.autograd.grad(loss, part, upstream[begin:end].contiguous())[0]
            gradient[begin:end].copy_(dx)
        return gradient, None, None, None, None
