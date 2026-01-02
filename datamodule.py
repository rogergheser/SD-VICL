from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import random
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class PairsInput:
    """Data structure for a pair of input images and their masks."""

    input_image: Image.Image
    guid_image: Image.Image
    guid_ground_truth: Image.Image
    ground_truth_mask: Image.Image
    input_category: str

    def to_tuple(
        self,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, str]:
        return (
            self.input_image,
            self.guid_image,
            self.guid_ground_truth,
            self.ground_truth_mask,
            self.input_category,
        )


class PairsDataset(Dataset):
    """Dataset for loading image pairs."""

    def __init__(
        self, root: str | Path, target_size: int | tuple[int, int], transform=None
    ):
        """
        Initialize the dataset.

        Args:
            input_images: List of input image file paths
            target_images: List of target image file paths
            transform: Optional transform to apply to images
        """
        self.root = Path(root)
        self.transform = transform
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        assert isinstance(target_size, tuple) and len(target_size) == 2, (
            "target_size must be an int or a tuple of two ints"
        )
        self.target_size = target_size
        self.input_images: list[str] = []
        self.target_images: list[str] = []
        self.samples = self.load_image_paths()
        self.flat_samples = self.get_flat_samples()
        print("Dataset initialized.")

    def get_flat_samples(self) -> list[dict[str, str]]:
        assert self.samples is not None
        flat_list: list[dict[str, str]] = []
        for key, category_samples in self.samples.items():
            assert isinstance(key, Path)
            category = key.stem
            for sample in category_samples:
                flat_list.append(sample.update({"category": str(category)}) or sample)
        return flat_list

    def load_image_paths(self) -> Mapping[Path, list[dict[str, str]]]:
        samples = defaultdict(list)
        for category in self.root.iterdir():
            if category.is_dir():
                for img_file in (category / "images").iterdir():
                    alpha_file = category / "alphas" / f"{img_file.stem}.png"
                    if not alpha_file.exists():
                        raise FileNotFoundError(f"Alpha file not found: {alpha_file}")
                    samples[category].append(
                        {
                            "image": str(img_file),
                            "mask": str(alpha_file),
                        }
                    )
        print(
            f"Loaded {sum(len(v) for v in samples.values())} samples from {self.root}"
        )
        return samples

    def __len__(self):
        """Return the number of samples in the dataset."""
        return sum([len(category) for category in self.samples.values()])

    def __getitem__(self, idx) -> PairsInput:
        """Get a sample from the dataset."""
        input_sample = self.flat_samples[idx]
        assert isinstance(self.target_size, tuple), "target_size must be set"
        input_image = (
            Image.open(input_sample["image"]).convert("RGB").resize(self.target_size)
        )
        ground_truth_mask = (
            Image.open(input_sample["mask"]).convert("RGB").resize(self.target_size)
        )
        input_category = input_sample["category"]

        remaining_categories = [
            category_path
            for category_path in self.samples.keys()
            if category_path.stem != input_category
        ]
        guid_category = random.choice(remaining_categories)
        guid_sample = random.choice(self.samples[guid_category])
        guid_image = (
            Image.open(guid_sample["image"]).convert("RGB").resize(self.target_size)
        )
        guid_ground_truth = (
            Image.open(guid_sample["mask"]).convert("RGB").resize(self.target_size)
        )

        if self.transform:
            input_image = self.transform(input_image)
            ground_truth_mask = self.transform(ground_truth_mask)

        return PairsInput(
            input_image=input_image,
            guid_image=guid_image,
            guid_ground_truth=guid_ground_truth,
            ground_truth_mask=ground_truth_mask,
            input_category=input_category,
        )
