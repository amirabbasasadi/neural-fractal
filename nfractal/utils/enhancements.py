import torch

def image_cdf(img):
    """
    computing the cumulative distribution function of an image
    input:
        img: image tensor
    """
    # calculate the histogram
    hist = torch.histc(img, bins=256, min=0, max=255)
    # normalize the histogram
    hist = hist.float() / hist.sum()
    # cumulative distribution function
    cdf = torch.cumsum(hist, 0)
    return cdf

def hist_equalizer(img, rescaling=True):
    """
    histogram equalization
    """
    # scaling the image
    if rescaling:
        img_min = img.min()
        img_scaled = (img-img_min)/(img.max()-img.min())
    else:
        img_scaled = img
    # calculate the cumulative distribution function
    cdf = image_cdf(img_scaled)
    img_scaled = torch.floor(img).to(torch.int64)
    equalized = cdf.index_select(0, img_scaled.view(-1))
    equalized = 255*equalized.reshape(img.shape)
    return equalized


