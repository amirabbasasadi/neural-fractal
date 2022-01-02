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
    def __init__(self, model, sample_size=2**17, img_size=(512, 512), threshold=2.0, center=(0.0, 0.0), zoom=1.0, batch_size=1,  device='cpu'):
        """
            Initialize Uniform Sampler
            Inputs:
                model: model to sample from, can be any PyTorch module
                sample_size: number of samples to generate at each iteration
                img_size: size of image to generate, a tuple (height, width)
                threshold: threshold to decide whether to keep a point
                zoom: zoom factor
                center: center of the image a tuple (x, y)
                batch_size: number of batches of samples, for cpu device 1 is recommended
                device: device to use, can be cpu or cuda
                verbose: whether to print progress
        """
        self.n = sample_size
        self.height = img_size[0]
        self.width = img_size[1]
        self.center_x = center[0]
        self.center_y = center[1]
        self.aspect_ratio = self.width/self.height
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.zoom = zoom
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
        # shifting and rescaling the points
        z.real = z.real + 0.5*self.zoom*self.aspect_ratio - self.center_x
        z.imag = z.imag + 0.5*self.zoom - self.center_y
        z = self.height*z/self.zoom

         # find corresponding pixels
        pixels = self.width*torch.floor(z.imag) + torch.floor(z.real)
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
                s.real.uniform_(self.center_x - 0.5*self.zoom*self.aspect_ratio,
                                self.center_x + 0.5*self.zoom*self.aspect_ratio)
                # sampling the imaginary part
                s.imag.uniform_(self.center_y - 0.5*self.zoom,
                                self.center_y + 0.5*self.zoom)
                # applying the dynamics iteratively
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
        # flipping the imaginary axis
        image = torch.flip(image, dims=[0])
        return image

      
