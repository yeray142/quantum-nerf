import numpy as np

from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.field_components.encodings import HashEncoding

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


class QuantumMLPWithHashEncoding(FieldComponent):
	"""A quantum MLP field component that uses TCNN for fast training."""

	def __init__(
		self,
		num_levels: int = 16,
		min_res: int = 16,
		max_res: int = 1024,
		log2_hashmap_size: int = 19,
		features_per_level: int = 2,
		hash_init_scale: float = 0.001,
		hidden_features: int = 8,
		hidden_layers: int = 3,
		out_features: int = 1,
		spectrum_layers = 4,
		use_noise = False,
		mlp_hidden_dim: int = 64,
		mlp_layers: int = 1,
	):
		super().__init__()

		self.num_levels = num_levels
		self.min_res = min_res
		self.max_res = max_res
		self.features_per_level = features_per_level
		self.hash_init_scale = hash_init_scale
		self.log2_hashmap_size = log2_hashmap_size
		self.hash_table_size = 2 ** log2_hashmap_size

		self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1

		# Hash Encoding parameters
		self.encoder = HashEncoding(
			num_levels=self.num_levels,
			min_res=self.min_res,
			max_res=self.max_res,
			log2_hashmap_size=self.log2_hashmap_size,
			features_per_level=self.features_per_level,
			hash_init_scale=self.hash_init_scale,
			implementation="torch",
		)

		# QIREN block
		self.qiren = Hybridren(
			in_features=self.encoder.get_out_dim(),
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

		mlp.append(nn.Linear(hidden_features, out_features))
		self.post_mlp = nn.Sequential(*mlp)

	def forward(self,x: Float[Tensor, "batch input_dims"]) -> Float[Tensor, "batch output_dims"]:
		"""
		Forward pass through the quantum MLP.
		"""
		enc_out = self.encoder(x)
		qiren_out, _ = self.qiren(enc_out)
		out = self.post_mlp(qiren_out)
		return out