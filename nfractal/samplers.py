import torch
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from nfractal.utils.dev import isnotebook

"""
    Base Sampler Class
"""
class Sampler():
    def sample(self):
        raise NotImplementedError("Sampler is an abstract class")

"""
    Uniform Sampler Class
    This class samples from complex plane uniformly
"""
class UniformSampler(Sampler):
    def __init__(self, model, sample_size=2**17, img_size=(512, 512), threshold=2.0, t=1.2, batch_size=1,  device='cpu'):
        """
            Initialize Uniform Sampler
            Inputs:
                model: model to sample from, can be any PyTorch module
                sample_size: number of samples to generate at each iteration
                img_size: size of image to generate, a tuple (height, width)
                threshold: threshold to decide whether to keep a point
                t: height of the complex plane to sample from
                batch_size: number of batches of samples, for cpu device 1 is recommended
                device: device to use, can be cpu or cuda
                verbose: whether to print progress
        """
        self.n = sample_size
        self.height = img_size[0]
        self.width = img_size[1]
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.t = t
        self.threshold = threshold
        self.device = device
        self.model = model.to(device)

        # check the environment
        self.notebook = isnotebook()

    def map_to_image(self, z):
        """
            Maps remaining points to the output image
            Inputs:
                z: complex tensor
            Outputs:
                image: image of the fractal
        """
        # shifting and rescale the points
        p = 0.5*(self.width + self.height*1.0j + (self.height/self.t)*z)
        # find corresponding pixels
        pixels = self.width*torch.floor(p.imag) + torch.floor(p.real)
        # counting the number of points in each pixel
        hist = torch.histc(pixels, bins=self.height*self.width, min=0, max=self.height*self.width-1)
        return hist
    
    def select_points(self, z, s):
        """
        Select remained points
        """
        return s[z.abs() < self.threshold]

    def sample(self, epochs=1, max_iters=50, verbose=False):
        # perform sampling in inference mode
        with torch.no_grad():
            # initial image
            image = torch.zeros(self.height*self.width).to(self.device)
            if verbose:
                if self.notebook:
                    epochs_bar = tqdm_notebook(total=epochs)
                else:
                    epochs_bar = tqdm(total=epochs)

            for epoch in range(epochs):
                s = torch.empty((self.batch_size, self.sample_size, 2)).to(self.device)
                z = torch.zeros((self.batch_size, self.sample_size, 2)).to(self.device)
                # treat tensors as complex
                z = torch.view_as_complex(z)
                s = torch.view_as_complex(s)
                # sampling the real part
                s.real.uniform_(-self.t*self.width/self.height,
                                self.t*self.width/self.height)
                # sampling the imaginary part
                s.imag.uniform_(-self.t, self.t)
                for i in range(max_iters):
                    if z.dim() < 3:
                        z = z.unsqueeze(2)
                    if s.dim() < 3:
                        s = s.unsqueeze(2)
                    z = self.model(z, s)

                z = z.view(-1)
                s = s.view(-1)
                # select points
                z = self.select_points(z, s)
                # map the selected points to image
                image += self.map_to_image(z)
                # print progress
                if verbose:
                    acc_rate = z.numel()/s.numel()
                    epochs_bar.update(1)
                    epochs_bar.set_description("Acceptance Rate {:.3f}".format(acc_rate))

        # reshaping to original image size
        image = image.view(self.height, self.width)
        # flipping the imaginary part
        image = torch.flip(image, dims=[0])
        return image

      
