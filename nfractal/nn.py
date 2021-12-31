import torch
import torch.nn as nn

class Linear(nn.Linear):
  """
  Complex Linear Transformation
  parameters are same as torch.nn.Linear
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for name, p in self.named_parameters():
      setattr(self, name,
              nn.Parameter(torch.complex(torch.randn_like(p),
                                         torch.randn_like(p))))

class ComplexActivation(nn.Module):
  """
  Generic Class for Complex Activations
  """
  def __init__(self, base_acitvation):
    super().__init__()
    self.activation = base_acitvation
  def forward(self, z):
    return torch.complex(self.activation(z.real),
                         self.activation(z.imag))
    