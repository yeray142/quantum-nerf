import pennylane as qml
import torch.nn as nn
import torch
import math
import numpy as np
from sympy.polys.densetools import dmp_integrate_in


#import qiskit.providers.aer.noise as noise


class FourierFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, learnable_features=False):
        super(FourierFeatures, self).__init__()
        frequency_matrix = torch.normal(mean=torch.zeros(out_channels, in_channels),
                                        std=1.0)
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        prefeatures = torch.einsum('oi,bli->blo', self.frequency_matrix.to(coordinates.device), coordinates)
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        return torch.cat((cos_features, sin_features), dim=2)


class QuantumLayer(nn.Module):
    def __init__(self, in_features, spectrum_layer, use_noise):
        super().__init__()

        self.in_features = in_features
        self.n_layer = spectrum_layer
        self.use_noise = use_noise

        def _circuit(inputs, weights1, weights2):
            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)
                for j in range(self.in_features):
                    qml.RZ(inputs[:, j], wires=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angle = np.pi + self.use_noise * np.random.rand()
                    qml.RX(rand_angle, wires=i)

            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            return res

        torch_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3),
                        "weights2": (2, in_features, 3)}
        self.qnode = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")
        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        
        org_type = x.dtype
        x_fp32 = x.to(torch.float32)
        out = self.qnn(x_fp32)
        if org_type != out.dtype:
            out = out.to(org_type)
        
        return out.reshape(orgin_shape)


class HybridLayer(nn.Module):
    def __init__(self, in_features, out_features, spectrum_layer, use_noise, bias=True, idx=0):
        super().__init__()
        self.idx = idx
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise)

    def forward(self, x):
        #print(f"HybridLayer {self.idx} input shape: {x.shape}")
        x1 = self.clayer(x)
        #print(f"HybridLayer {self.idx} after clayer shape: {x1.shape}")
        # x1 = self.norm(x1.permute(0, 2, 1)).permute(0, 2, 1)
        x1 = self.norm(x1)
        out = self.qlayer(x1)
        return out


class Hybridren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise,
                 outermost_linear=True):
        super().__init__()

        self.net = []
        self.net.append(HybridLayer(in_features, hidden_features, spectrum_layer, use_noise, idx=1))

        for i in range(hidden_layers):
            self.net.append(HybridLayer(hidden_features, hidden_features, spectrum_layer, use_noise, idx=i + 2))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
        else:
            final_linear = HybridLayer(hidden_features, out_features, spectrum_layer, use_noise)

        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

class FGScalingLayer(nn.Module):
    """
    FourierGaussian-based scaling layer that projects input coordinates onto weighted Fourier bases.

    The layer:
    1. Repeats input x ∈ R^{d_in} n times to get x_rep ∈ R^{n*d_in}
    2. Creates Fourier basis matrix B with elements b_{k,j} = cos(w_f * s_j + φ_p)
    3. Projects through weighted bases: h_1 = Λ * B * x_rep + b

    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension
        output_dim (int): Dimension of the final output after linear transformation
        n (int): Number of repetitions for input concatenation
        F (int): Number of frequencies
        P (int): Number of phases per frequency
        gamma (float): Gaussian shaping parameter
        sampling_range (tuple): Range for sampling s, default (-2π, 2π)
    """
    def __init__(self, d_in, d_out, output_dim, n=4, F=8, P=4, gamma=1.0, sampling_range=(-2 * np.pi, 2 * np.pi)):
        super(FGScalingLayer, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.output_dim = output_dim
        self.n = n
        self.F = F
        self.P = P
        self.gamma = gamma

        # Dimension of the concatenated input
        self.nd_in = d_in * n

        # Create a frequency array w_f
        self.w_f = torch.arange(1, F + 1, dtype=torch.float32)

        # Create a phase array φ_p distributed uniformly in [0, 2π)
        self.phi_p = torch.linspace(0, 2 * np.pi * (P - 1) / P, P, dtype=torch.float32)

        # Create sampling points s
        self.s_j = torch.linspace(-2 * math.pi, 2 * math.pi, self.nd_in, dtype=torch.float32)

        # Initialize fixed coefficient matrix Λ ∈ R^{d_out × (F×P)}
        self.Lambda = nn.Parameter(
            torch.randn(d_out, F * P) * 0.1,
            requires_grad=False # fixed
        )

        # Trainable bias parameter
        self.bias = nn.Parameter(torch.zeros(d_out))

        # Pre-compute Fourier basis matrix B
        self._build_fourier_basis()

        # Linear layer for frequency spectrum adjustment towards target domain
        self.linear = nn.Linear(d_out, output_dim)

        # BatchNorm for training stability and convergence acceleration
        self.batch_norm = nn.BatchNorm1d(output_dim)


    def _build_fourier_basis(self):
        """
        Build the Fourier basis matrix B with elements:
        b_{k,j} = cos(w_f * s_j + φ_p) where k = (f-1) × P + p
        """
        # Initialize basis matrix B ∈ R^{(F×P) × nd_in}
        B = torch.zeros(self.F * self.P, self.nd_in)

        # Iterate over frequencies and phases
        for f in range(self.F):
            for p in range(self.P):
                k = f * self.P + p  # k = (f-1) × P + p (adjusted for 0-indexing)

                # Compute b_{k,j} = cos(w_f * s_j + φ_p)
                w_f = self.w_f[f]
                phi_p = self.phi_p[p]

                # Broadcasting: w_f * s_j + phi_p for all j
                B[k, :] = torch.cos(w_f * self.s_j + phi_p)

        # Register as buffer (non-trainable but part of state_dict)
        self.register_buffer('B', B)

    def forward(self, x):
        """
        Forward pass of the Fourier scaling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_in)

        Returns:
            torch.Tensor: Output tensor h_2 of shape (batch_size, d_out)
        """
        batch_size = x.shape[0]

        # Step 1: Repeat input n times with concatenation
        # x_rep ∈ R^{nd_in}
        x_rep = x.repeat(1, self.n)  # Shape: (batch_size, nd_in)

        # Step 2: Project to Fourier bases
        # Compute W = ΛB (Fourier-composed filter)
        W = torch.mm(self.Lambda, self.B)  # Shape: (d_out, nd_in)

        # Step 3: Linear transformation h_1 = ΛBx_rep + b
        h_1 = torch.mm(x_rep, W.T) + self.bias  # Shape: (batch_size, d_out)

        # Step 4: Apply Gaussian weighting
        # ε = exp(-γh_1^2)
        epsilon = torch.exp(-self.gamma * h_1.pow(2))  # Shape: (batch_size, d_out)

        # Step 5: Final output h_2 = εh_1
        h_2 = epsilon * h_1  # Shape: (batch_size, d_out)

        # Step 6: Apply linear layer and batch normalization
        linear_out = self.batch_norm(self.linear(h_2)) # Shape: (batch_size, output_dim)

        return linear_out

    def get_fourier_filter(self):
        """
        Get the computed Fourier-composed filter W = ΛB.

        Returns:
            torch.Tensor: Fourier filter of shape (d_out, nd_in)
        """
        return torch.mm(self.Lambda, self.B)

    def extra_repr(self):
        """String representation of the layer."""
        return (f'd_in={self.d_in}, d_out={self.d_out}, n={self.n}, '
                f'F={self.F}, P={self.P}, gamma={self.gamma}')

class QFGN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, spectrum_layer=1, use_noise=0,
                 outermost_linear=True):
        super().__init__()

        # Initialize the Fourier Gaussian scaling
        self.scaling = FGScalingLayer(
            d_in=in_features,
            d_out=hidden_features,
            output_dim=out_features,
            n=4,  # Number of repetitions
            F=8,  # Number of frequencies
            P=4,  # Number of phases per frequency
            gamma=0.8,  # Gaussian shaping parameter
        )

        # Initialize the quantum layer
        self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise)

        # If outermost_linear is True, add a final linear layer
        if outermost_linear:
            self.qlayer = nn.Sequential(
                self.qlayer,
                nn.Linear(out_features, out_features)
            )

    def forward(self, coords):
        output = self.scaling(coords)
        output = self.qlayer(output)
        return output, coords


class FFNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, res=True):
        super().__init__()
        self.res = res
        self.clayer1 = nn.Linear(in_features, 2 * in_features, bias=bias)
        self.norm = nn.BatchNorm1d(2 * in_features)
        self.activ = nn.ReLU()
        self.clayer2 = nn.Linear(2 * in_features, out_features, bias=bias)

    def forward(self, x):
        x1 = self.clayer1(x)
        x1 = self.norm(x1.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.clayer2(self.activ(x1))
        if self.res:
            out = out + x
        return out


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        self.net = []
        self.net.append(FFNLayer(in_features, hidden_features, res=False))

        for i in range(hidden_layers):
            self.net.append(FFNLayer(hidden_features, hidden_features))

        self.net.append(FFNLayer(hidden_features, out_features, res=False))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


class SineLayer_bn(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, activ='relu', omega_0=30):
        super().__init__()

        self.is_first = is_first
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'sine':
            self.activ = torch.sin

    def forward(self, input):
        x1 = self.linear(input)
        x1 = self.omega_0 * self.norm(x1.permute(0, 2, 1)).permute(0, 2, 1)
        return self.activ(x1)


class Siren_bn(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, activ='relu',
                 first_omega_0=30, hidden_omega_0=30, rff=False):
        super().__init__()

        self.net = []
        if rff:
            self.net.append(FourierFeatures(in_features, hidden_features // 2))
        else:
            self.net.append(
                SineLayer_bn(in_features, hidden_features, is_first=True, activ=activ, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(
                SineLayer_bn(hidden_features, hidden_features, is_first=False, activ=activ, omega_0=hidden_omega_0))

        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(SineLayer_bn(hidden_features, out_features, is_first=False, activ=activ))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, idx=0):
        super().__init__()
        self.idx = idx
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = self.omega_0 * self.linear(input)
        return torch.sin(out)

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, idx=1))

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, idx=i + 2))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
