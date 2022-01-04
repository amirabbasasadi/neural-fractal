import torch
from nfractal.utils.colorspace import rgb_to_lab, lab_to_rgb

class ColorTransfer():
    """
    ColorTransfer class
    """
    def __init__(self, target_img):
        """
        Initialize ColorTransfer class
        input:
            target_img: target image
        """
        t_img = target_img[:, :, :3].to(torch.float64)
        # resclae to [0, 1]
        if t_img.max() > 1.01:
            t_img = t_img / 255.0
        t_img_lab = rgb_to_lab(t_img).permute(2, 0, 1).flatten(1)
        t_sigma, t_mu = torch.std_mean(t_img_lab, dim=1, keepdim=True)
        # keeping the information of target color space
        self.t_sigma = t_sigma
        self.t_mu = t_mu
    
    def __call__(self, source_img):
        """
        Transfer color from source image to target image
        input:
            source_img: source image
        output:
            source_img: source image with transferred color
        """
        if source_img.dim() == 2:
            s_img = source_img.unsqueeze(2)
            s_img = s_img.repeat(1, 1, 3)
        else:
            s_img = source_img
        s_img = s_img[:, :, :3].to(torch.float64)
        if s_img.max() > 1.01:
            s_img = s_img / 255.0
        s_img_lab = rgb_to_lab(s_img).permute(2, 0, 1).flatten(1)
        s_sigma, s_mu = torch.std_mean(s_img_lab, dim=1, keepdim=True)
        # normalize source color space
        s_img_lab = (s_img_lab - s_mu) / s_sigma
        # transfer color
        s_img_lab = s_img_lab * self.t_sigma + self.t_mu
        result_lab = s_img_lab.unflatten(1, (source_img.shape[0], source_img.shape[1]))
        result_lab = result_lab.permute(1, 2, 0)
        # convert back to rgb
        result_rgb = lab_to_rgb(result_lab)
        # clamping the result
        result_rgb = torch.clamp(result_rgb, 0.0, 1.0)
        # reshapeing image to original shape
        return result_rgb