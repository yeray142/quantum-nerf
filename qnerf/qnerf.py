"""
TODO: Complete this file with your own model.
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model

from qnerf.qnerf_field import QuantumNerfField


@dataclass
class QNerfModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: QNerfModel)

    hidden_dim = 8
    hidden_dim_color = 8

    use_appearance_embedding = False
    appearance_embed_dim = 32



class QNerfModel(NerfactoModel):
    """
    Quantum Nerf Model.

    Args:
    	config: QNerfModelConfig
			Configuration for the model.
    """

    config: QNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        self.field = QuantumNerfField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            hidden_dim=self.config.hidden_dim,
            hidden_dim_color=self.config.hidden_dim_color,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.