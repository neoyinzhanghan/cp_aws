import os
import timm
import torch.nn as nn
import torch
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms.functional import to_pil_image
from huggingface_hub import login

os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

login(
    token="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"
)  # login with your User Access Token, found at https://huggingface.co/settings/tokens


def adjust_transformations(transform_pipeline):
    new_transforms = []
    for transform in transform_pipeline.transforms:
        # Skip adding F.to_tensor to the new pipeline
        if isinstance(transform, transforms.ToTensor):
            continue
        new_transforms.append(transform)
    return transforms.Compose(new_transforms)


class UNIExtractor(nn.Module):
    """A wrapper model for UNI to extract features from images"""

    def __init__(self):
        super().__init__()
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        # Usage:
        original_transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        self.transform = adjust_transformations(original_transform)

        self.model.eval()

    def forward(self, x):
        # outputs = []
        # for img in x:
        #     img_pil = to_pil_image(img)
        #     transformed = self.transform(img_pil)
        #     transformed_tensor = transformed.to(device=x.device, dtype=x.dtype).unsqueeze(0)  # Keep batch dimension
        #     outputs.append(transformed_tensor)
        # x_transformed = torch.cat(outputs, dim=0)  # Concatenate along the batch dimension
        x_transformed = self.transform(x)
        return self.model(x_transformed)


def load_model():
    # pretrained=True needed to load UNI weights (and download weights for the first time)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)

    # load the UNIExtractor model
    model = UNIExtractor()

    return model
