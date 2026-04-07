import torch
import torch.nn as nn
class Denoiser(nn.Module):
    def __init__(self, shared_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(shared_dim * 2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim)
        )

    def forward(self, x_t, t, cond):
        t = t.unsqueeze(1).float()
        inp = torch.cat([x_t, cond, t], dim=1)
        return self.net(inp)
    
def diffusion_loss(denoiser, x0, cond, T=1000):
    device = x0.device

    t = torch.randint(0, T, (x0.size(0),), device=device)

    noise = torch.randn_like(x0)

    alpha_t = 0.9  # simplified schedule
    x_t = torch.sqrt(torch.tensor(alpha_t)) * x0 + \
        torch.sqrt(torch.tensor(1 - alpha_t)) * noise

    noise_pred = denoiser(x_t, t, cond)

    return torch.mean((noise - noise_pred) ** 2)