import torch

def inject_trigger(x, pattern, mask):
    """
    blending function B(x,t) = x * (1- m)+ p * m

    where:
        - x: clean images (B, C, H, W)
        - t: target class
        - p: generated pattern, (B, C, H, W), values in [0, 1])
        - m: blending mask, (1, 1, H, W), values in {0, 1}

    returns the poisoned image x'
    """

    return x * (1.0 - mask)+ pattern * mask

def diversity_loss(x, x_prime, gx, gx_prime):
    """
    L_div = ||x - x'|| / ||g(x) - g(x')||
 
    """
    input_diff = torch.norm(x.flatten(1) - x_prime.flatten(1), dim =1)
    trigger_diff = torch.norm(gx.flatten(1) - gx_prime.flatten(1), dim =1)
    return torch.mean(input_diff / (trigger_diff + 1e-8))

