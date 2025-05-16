"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Optional, Tuple, Dict

from jaxtyping import Shaped, Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox

import torch
from nerfstudio.field_components.field_heads import FieldHeadNames
from torch import Tensor
from torch.nn import Embedding

from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions  # for custom Field

from qnerf.modules.modules import Hybridren


class QuantumNerfField(Field):
    """Quantum Nerf field.

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        spectrum_layers: number of quantum reuploading layers
        hidden_dim: number of hidden dimensions
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of color layers
        spectrum_layers_color: number of quantum reuploading layers for color
        hidden_dim_color: number of color hidden dimensions
        appearance_embedding_dim: appearance embedding dimension
        spatial_distortion: spatial distortion to apply to the field

    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        spectrum_layers: int = 3,
        hidden_dim: int = 8,
        geo_feat_dim: int = 4, # TODO: No idea if this is correct
        num_layers_color: int = 3,
        spectrum_layers_color: int = 3,
        hidden_dim_color: int = 8,

        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = False,

        spatial_distortion: Optional[SpatialDistortion] = None
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None
        self.use_average_appearance_embedding = use_average_appearance_embedding

        self.mlp_base = Hybridren(
            in_features=3, # TODO: No idea if this is correct
            hidden_features=hidden_dim,
            hidden_layers=num_layers,
            out_features=1 + self.geo_feat_dim,
            spectrum_layer=spectrum_layers,
            use_noise=False,
            outermost_linear=True,
        )

        self.mlp_head = Hybridren(
            in_features=3 + self.geo_feat_dim + self.appearance_embedding_dim,
            hidden_features=hidden_dim_color,
            hidden_layers=num_layers_color,
            out_features=3,
            spectrum_layer=spectrum_layers_color,
            use_noise=False,
            outermost_linear=True,
        )

    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        """
        Get the density of the field.
        Args:
            ray_samples: RaySamples object containing the ray samples.

        Returns:
            Tuple containing the density and the geometry features.
        """
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]

        assert positions.numel() > 0, "No positions to sample"

        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        assert positions_flat.numel() > 0, "No positions to sample"
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_after_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None

        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Ray samples must have camera indices")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # TODO: (Optional) Add appearance embedding to the field
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        h = torch.cat(
            [
                directions_flat,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
            ),
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs