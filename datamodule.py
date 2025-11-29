"""
HuggingFace Dataset Module for SD-VICL

This module provides datamodule classes for loading datasets from HuggingFace Hub,
specifically designed for visual in-context learning experiments.

Usage:
    from datamodule import CoCoDataModule

    # Load COCO dataset
    datamodule = CoCoDataModule(split="train")
    
    # Get a sample
    sample = datamodule[0]
    print(sample["image"])  # PIL Image
    print(sample["objects"])  # Object annotations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataModuleConfig:
    """Configuration for datamodule."""
    dataset_name: str
    split: str = "train"
    cache_dir: Optional[str] = None
    streaming: bool = False
    trust_remote_code: bool = False
    transform: Optional[Callable[[Image.Image], Any]] = None
    target_transform: Optional[Callable[[Any], Any]] = None


class HuggingFaceDataModule(ABC, Dataset):
    """
    Abstract base class for HuggingFace dataset modules.
    
    This class provides a common interface for loading and processing
    datasets from the HuggingFace Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        split: Dataset split to load (e.g., "train", "validation", "test")
        cache_dir: Directory to cache downloaded datasets
        streaming: Whether to stream the dataset instead of downloading
        trust_remote_code: Whether to trust remote code for custom datasets
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to targets/annotations
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        trust_remote_code: bool = False,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ):
        """Initialize the HuggingFace datamodule."""
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        self.transform = transform
        self.target_transform = target_transform
        
        self._dataset = None
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the dataset from HuggingFace Hub."""
        from datasets import load_dataset
        
        self._dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )
    
    @property
    def dataset(self):
        """Get the underlying HuggingFace dataset."""
        return self._dataset
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.streaming:
            raise ValueError("Cannot determine length of streaming dataset")
        return len(self._dataset)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        pass
    
    def get_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for the dataset.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            collate_fn: Function to collate samples into a batch
            **kwargs: Additional arguments passed to DataLoader
            
        Returns:
            PyTorch DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )


@dataclass
class CoCoConfig:
    """Configuration specific to COCO dataset."""
    dataset_name: str = "detection-datasets/coco"
    split: str = "train"
    cache_dir: Optional[str] = None
    streaming: bool = False
    trust_remote_code: bool = False
    transform: Optional[Callable[[Image.Image], Any]] = None
    target_transform: Optional[Callable[[Any], Any]] = None
    return_masks: bool = False
    image_size: Optional[int] = None


class CoCoDataModule(HuggingFaceDataModule):
    """
    DataModule for loading COCO dataset from HuggingFace Hub.
    
    The COCO (Common Objects in Context) dataset is a large-scale object detection,
    segmentation, and captioning dataset. This module provides easy access to COCO
    through the HuggingFace Datasets library.
    
    Available splits:
        - train: Training set
        - val: Validation set
    
    Each sample contains:
        - image: PIL Image
        - image_id: Unique identifier for the image
        - width: Image width
        - height: Image height
        - objects: Dictionary containing:
            - id: List of object IDs
            - area: List of object areas
            - bbox: List of bounding boxes [x, y, width, height]
            - category: List of category IDs
    
    Example:
        >>> from datamodule import CoCoDataModule
        >>> 
        >>> # Load COCO training set
        >>> coco = CoCoDataModule(split="train")
        >>> 
        >>> # Access a sample
        >>> sample = coco[0]
        >>> image = sample["image"]  # PIL Image
        >>> objects = sample["objects"]  # Annotations
        >>> 
        >>> # Create a dataloader
        >>> dataloader = coco.get_dataloader(batch_size=4, shuffle=True)
    
    Args:
        dataset_name: HuggingFace dataset name (default: "detection-datasets/coco")
        split: Dataset split ("train" or "val")
        cache_dir: Directory to cache the dataset
        streaming: Whether to stream instead of downloading
        trust_remote_code: Whether to trust remote code
        transform: Transform to apply to images
        target_transform: Transform to apply to annotations
        return_masks: Whether to include segmentation masks (if available)
        image_size: Optional size to resize images to (maintains aspect ratio)
    """
    
    def __init__(
        self,
        dataset_name: str = "detection-datasets/coco",
        split: str = "train",
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        trust_remote_code: bool = False,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        return_masks: bool = False,
        image_size: Optional[int] = None,
    ):
        """Initialize the COCO datamodule."""
        self.return_masks = return_masks
        self.image_size = image_size
        
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
            transform=transform,
            target_transform=target_transform,
        )
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized PIL Image
        """
        if self.image_size is None:
            return image
        
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = self.image_size
            new_height = int(height * self.image_size / width)
        else:
            new_height = self.image_size
            new_width = int(width * self.image_size / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the COCO dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - image: PIL Image (or transformed version)
                - image_id: Image identifier
                - width: Original image width
                - height: Original image height
                - objects: Annotation data (bboxes, categories, etc.)
        """
        sample = self._dataset[idx]
        
        # Extract image
        image = sample.get("image")
        if image is None:
            raise KeyError(f"Sample at index {idx} does not contain an 'image' field")
        
        # Convert to PIL Image if necessary
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Capture original dimensions before any transforms
        original_width = sample.get("width", image.width)
        original_height = sample.get("height", image.height)
        
        # Resize if specified
        if self.image_size is not None:
            image = self._resize_image(image)
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Build result dictionary
        result = {
            "image": image,
            "image_id": sample.get("image_id", idx),
            "width": original_width,
            "height": original_height,
        }
        
        # Extract objects/annotations
        objects = sample.get("objects", {})
        
        # Apply target transform if provided
        if self.target_transform is not None:
            objects = self.target_transform(objects)
        
        result["objects"] = objects
        
        return result
    
    def get_category_names(self) -> Optional[List[str]]:
        """
        Get the list of category names if available.
        
        Returns:
            List of category names or None if not available
        """
        if hasattr(self._dataset, "features"):
            features = self._dataset.features
            if "objects" in features and hasattr(features["objects"], "feature"):
                obj_feature = features["objects"].feature
                if "category" in obj_feature and hasattr(obj_feature["category"], "names"):
                    return obj_feature["category"].names
        return None
    
    def get_sample_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a sample by its image ID.
        
        Note: This performs a linear search through the dataset with O(n) complexity.
        For large datasets, consider building an index mapping image_id to idx
        if frequent lookups by ID are needed.
        
        Args:
            image_id: The image ID to search for
            
        Returns:
            Sample dictionary or None if not found
        """
        if self.streaming:
            raise ValueError("Cannot search by ID in streaming mode")
        
        for idx in range(len(self)):
            sample = self._dataset[idx]
            if sample.get("image_id") == image_id:
                return self[idx]
        return None


def create_coco_datamodule(
    split: str = "train",
    cache_dir: Optional[str] = None,
    transform: Optional[Callable[[Image.Image], Any]] = None,
    image_size: Optional[int] = None,
    **kwargs
) -> CoCoDataModule:
    """
    Factory function to create a COCO datamodule with common defaults.
    
    Args:
        split: Dataset split ("train" or "val")
        cache_dir: Directory to cache the dataset
        transform: Transform to apply to images
        image_size: Optional size to resize images to
        **kwargs: Additional arguments passed to CoCoDataModule
        
    Returns:
        Configured CoCoDataModule instance
    
    Example:
        >>> coco = create_coco_datamodule(split="train", image_size=512)
        >>> sample = coco[0]
    """
    return CoCoDataModule(
        split=split,
        cache_dir=cache_dir,
        transform=transform,
        image_size=image_size,
        **kwargs
    )
