import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()
    def forward(self, x, freq=25.):
        return torch.sin(freq * x)

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()
        
        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='sin', film_exp_freq=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.film_exp_freq = film_exp_freq

    def forward(self, x, freq, phase_shift=None,random_phase=False):
        x = self.layer(x)
        if not freq.shape == x.shape:
            freq = freq.unsqueeze(1).expand_as(x)
        if not phase_shift is None and not phase_shift.shape == x.shape:
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        if phase_shift is None:
            phase_shift = 0

        if self.film_exp_freq:
            freq = torch.exp(freq)

        if self.activation == 'sin':
            if random_phase:
                phase_shift = phase_shift*torch.randn(x.shape[0],x.shape[1],1).to(x.device)
            return torch.sin(freq * x + phase_shift)
        else:
            return F.leaky_relu(freq * x + phase_shift, negative_slope=0.2)

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor
    
    def forward_var(self, variances):
        return variances * self.scale_factor**2
    
    def forward_inv(self,coordinates):
        return coordinates / self.scale_factor

#------------------------------------------------------------------------------------------------------------
## deformation network
class SPATIAL_SIREN_DEFORM(nn.Module):
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None, phase_noise=False, hidden_z_dim=256,  **kwargs):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.phase_noise = phase_noise
        self.film_exp_freq = kwargs.get('film_exp_freq', False)

        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim, film_exp_freq=self.film_exp_freq),
            FiLMLayer(hidden_dim, hidden_dim, film_exp_freq=self.film_exp_freq),
            FiLMLayer(hidden_dim, hidden_dim, film_exp_freq=self.film_exp_freq),
            FiLMLayer(hidden_dim, hidden_dim, film_exp_freq=self.film_exp_freq),
        ])

        self.network.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        self.final_layer = nn.Linear(hidden_dim, 4)
        self.final_layer.apply(frequency_init(25))

        nn.init.zeros_(self.final_layer.weight)  # prevent large deformation at the start of training
        self.mapping_network = CustomMappingNetwork(z_dim, hidden_z_dim, len(self.network)*hidden_dim*2)

        self.gridwarper = None
        if 'point_normalize' in kwargs.keys() and float(kwargs['point_normalize']) != 2.0:
            self.gridwarper = UniformBoxWarp(float(kwargs['point_normalize']))

        self.correction_scale = 1.0
        if 'correct_scale' in kwargs.keys() and float(kwargs['correct_scale']) != 1.0:
            self.correction_scale = float(kwargs['correct_scale']) 

        self.deformation_scale = 1.0
        if 'deform_scale' in kwargs.keys() and float(kwargs['deform_scale']) != 1.0:
            self.deformation_scale = float(kwargs['deform_scale']) 


    def forward(self, input, z, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, **kwargs):
        frequencies = frequencies*15 + 30
        
        if self.gridwarper is not None:
            input = self.gridwarper(input)
        
        x = input
        
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end],random_phase=self.phase_noise)
        
        x = self.final_layer(x)
        dx, dsigma = x[...,:3], x[...,3:4]

        if self.correction_scale != 1.0:
            dsigma = dsigma * self.correction_scale

        if self.deformation_scale != 1.0:
            dx = dx * self.deformation_scale
        return dx, dsigma

#-----------------------------------------------------------------------------------------------------------
## template network
class SINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='sin'):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation

    def forward(self, x, freq=30., phase_shift=None,random_phase=False):
        x = self.layer(x)
        if phase_shift is None:
            phase_shift = 0
        if self.activation == 'sin':
            return torch.sin(freq * x + phase_shift)

class SPATIAL_SIREN_TEMPLATE_DEFERDIR(nn.Module):
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None, phase_noise=False, hidden_z_dim=256, **kwargs):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.phase_noise = phase_noise
        self.film_exp_freq = kwargs.get('film_exp_freq', False)
        
        self.network = nn.ModuleList([
            SINLayer(3, hidden_dim),
            SINLayer(hidden_dim, hidden_dim),
            SINLayer(hidden_dim, hidden_dim),
            SINLayer(hidden_dim, hidden_dim),
            SINLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_network = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim, film_exp_freq=self.film_exp_freq),
            FiLMLayer(hidden_dim, hidden_dim, film_exp_freq=self.film_exp_freq),
            FiLMLayer(hidden_dim, hidden_dim, film_exp_freq=self.film_exp_freq),
            FiLMLayer(hidden_dim+3, hidden_dim, film_exp_freq=self.film_exp_freq),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))        
        self.mapping_network = CustomMappingNetwork(z_dim, hidden_z_dim, len(self.color_network)*hidden_dim*2)

        # -- initialization         
        self.network.apply(sine_init)
        self.network[0].apply(first_layer_sine_init)
        self.final_layer.apply(frequency_init(30))

        self.color_network.apply(frequency_init(25))
        self.color_network[0].apply(first_layer_film_sine_init)
        self.color_layer_linear.apply(frequency_init(25))
        
        self.gridwarper = None
        if 'point_normalize' in kwargs.keys() and float(kwargs['point_normalize']) != 2.0:
            self.gridwarper = UniformBoxWarp(float(kwargs['point_normalize']))

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        x = input

        if self.gridwarper is not None:
            x = self.gridwarper(x)

        if kwargs.get('fix_density_layers', False):  
            with torch.no_grad():
                for layer in self.network:
                    x = layer(x)
                sigma = self.final_layer(x)
        else:
            for layer in self.network:
                x = layer(x)
            sigma = self.final_layer(x)


        for index, layer in enumerate(self.color_network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim

            if index == len(self.color_network)-1 :
                x = torch.cat([ray_directions, x], dim=-1)
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end], random_phase=self.phase_noise)

        rbg = torch.sigmoid(self.color_layer_linear(x))
        return torch.cat([rbg, sigma], dim=-1)
