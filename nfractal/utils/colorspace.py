import torch

# a set of transformations to change the color space
rgb_to_lms_mat = torch.tensor([[0.3811, 0.5783, 0.0402],
                               [0.1967, 0.7244, 0.0782],
                               [0.0241, 0.1288, 0.8444]], dtype=torch.float64)

lms_to_rgb_mat = torch.linalg.inv(rgb_to_lms_mat)

lms_to_lab_mat = torch.tensor([[ 0.5774,  0.5774,  0.5774],
                               [ 0.4082,  0.4082, -0.8165],
                               [ 0.7071, -0.7071,  0.0000]], dtype=torch.float64)

lab_to_lms_mat = torch.linalg.inv(lms_to_lab_mat)

def rgb_to_lab(rgb):
  """
    Convert rgb to lab
    input:
        rgb: rgb image
    output:
        lab: lab image
  """
  lms = torch.log1p(rgb_to_lms_mat @ rgb.permute(2, 0, 1).flatten(1))
  lab = lms_to_lab_mat @ lms
  lab = lab.unflatten(1, (rgb.shape[0], rgb.shape[1])).permute(1, 2, 0)
  return lab

"""
    Convert rgb to lab
    input:
        rgb: rgb image
    output:
        lab: lab image
"""
def lab_to_rgb(lab):
  lms = torch.expm1(lab_to_lms_mat @ lab.permute(2, 0, 1).flatten(1))
  rgb = lms_to_rgb_mat @ lms
  rgb = rgb.unflatten(1, (lab.shape[0], lab.shape[1])).permute(1, 2, 0)
  return rgb
