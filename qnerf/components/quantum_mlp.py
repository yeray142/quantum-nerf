from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent

from qnerf.components.modules import Hybridren


class QuantumMLP(FieldComponent):
	"""A quantum MLP field component that uses TCNN for fast training."""

	def __init__(
		self,
		in_features: int = 3,
		hidden_features: int = 8,
		hidden_layers: int = 3,
		out_features: int = 1,
		spectrum_layers = 4,
		use_noise = False,
		mlp_hidden_dim: int = 64,
		mlp_layers: int = 2,
	):
		super().__init__()

		# QIREN block
		self.qiren = Hybridren(
			in_features=in_features,
			hidden_features=hidden_features,
			hidden_layers=hidden_layers,
			out_features=hidden_features,
			spectrum_layer=spectrum_layers,
			use_noise=use_noise,
			outermost_linear=True
		)

		# Classical MLP after QIREN
		mlp = []
		for _ in range(mlp_layers - 1):
			mlp.append(nn.Linear(hidden_features, mlp_hidden_dim))
			mlp.append(nn.ReLU())
			hidden_features = mlp_hidden_dim

		mlp.append(nn.Linear(mlp_hidden_dim, out_features))
		self.post_mlp = nn.Sequential(*mlp)

	def forward(self,x: Float[Tensor, "batch input_dims"]) -> Float[Tensor, "batch output_dims"]:
		"""
		Forward pass through the quantum MLP.
		"""
		qiren_out, _ = self.qiren(x)
		out = self.post_mlp(qiren_out)
		return out