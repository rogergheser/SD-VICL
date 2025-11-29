"""
Tests for the datamodule.

These tests verify the datamodule functionality using mocked datasets
since network access to HuggingFace Hub may not be available.
"""

import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from datamodule import (
    HuggingFaceDataModule,
    CoCoDataModule,
    CoCoConfig,
    DataModuleConfig,
    create_coco_datamodule,
)

# Correct patch target - patches within the datasets module
LOAD_DATASET_PATCH = 'datasets.load_dataset'


def create_mock_coco_sample(image_id: int = 0):
    """Create a mock COCO sample for testing."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    return {
        "image": image,
        "image_id": image_id,
        "width": 640,
        "height": 480,
        "objects": {
            "id": [1, 2, 3],
            "area": [1000, 2000, 1500],
            "bbox": [[10, 20, 100, 150], [200, 100, 50, 80], [300, 200, 120, 90]],
            "category": [1, 2, 1],
        }
    }


def create_mock_dataset(num_samples: int = 10):
    """Create a mock dataset for testing."""
    mock_dataset = MagicMock()
    samples = [create_mock_coco_sample(i) for i in range(num_samples)]
    mock_dataset.__len__ = MagicMock(return_value=num_samples)
    mock_dataset.__getitem__ = MagicMock(side_effect=lambda idx: samples[idx])
    return mock_dataset


class TestCoCoDataModule(unittest.TestCase):
    """Test cases for CoCoDataModule."""
    
    @patch(LOAD_DATASET_PATCH)
    def test_initialization(self, mock_load_dataset):
        """Test that CoCoDataModule initializes correctly."""
        mock_load_dataset.return_value = create_mock_dataset()
        
        datamodule = CoCoDataModule(split="train")
        
        self.assertEqual(datamodule.dataset_name, "detection-datasets/coco")
        self.assertEqual(datamodule.split, "train")
        self.assertFalse(datamodule.streaming)
        mock_load_dataset.assert_called_once()
    
    @patch(LOAD_DATASET_PATCH)
    def test_len(self, mock_load_dataset):
        """Test dataset length."""
        mock_load_dataset.return_value = create_mock_dataset(10)
        
        datamodule = CoCoDataModule(split="train")
        
        self.assertEqual(len(datamodule), 10)
    
    @patch(LOAD_DATASET_PATCH)
    def test_getitem(self, mock_load_dataset):
        """Test getting a sample from the dataset."""
        mock_load_dataset.return_value = create_mock_dataset(5)
        
        datamodule = CoCoDataModule(split="train")
        sample = datamodule[0]
        
        self.assertIn("image", sample)
        self.assertIn("image_id", sample)
        self.assertIn("objects", sample)
        self.assertIsInstance(sample["image"], Image.Image)
    
    @patch(LOAD_DATASET_PATCH)
    def test_image_resize(self, mock_load_dataset):
        """Test image resizing functionality."""
        mock_load_dataset.return_value = create_mock_dataset(1)
        
        datamodule = CoCoDataModule(split="train", image_size=256)
        sample = datamodule[0]
        
        image = sample["image"]
        # Check that the larger dimension is 256
        self.assertTrue(max(image.size) == 256)
    
    @patch(LOAD_DATASET_PATCH)
    def test_transform(self, mock_load_dataset):
        """Test custom transform application."""
        mock_load_dataset.return_value = create_mock_dataset(1)
        
        # Simple transform that returns image size
        transform = lambda img: img.size
        
        datamodule = CoCoDataModule(split="train", transform=transform)
        sample = datamodule[0]
        
        # The image should now be a tuple (width, height)
        self.assertIsInstance(sample["image"], tuple)
    
    @patch(LOAD_DATASET_PATCH)
    def test_target_transform(self, mock_load_dataset):
        """Test custom target transform application."""
        mock_load_dataset.return_value = create_mock_dataset(1)
        
        # Transform that returns number of objects
        target_transform = lambda obj: len(obj.get("id", []))
        
        datamodule = CoCoDataModule(split="train", target_transform=target_transform)
        sample = datamodule[0]
        
        # Objects should now be the count
        self.assertEqual(sample["objects"], 3)
    
    @patch(LOAD_DATASET_PATCH)
    def test_dataloader(self, mock_load_dataset):
        """Test dataloader creation."""
        mock_load_dataset.return_value = create_mock_dataset(4)
        
        datamodule = CoCoDataModule(split="train")
        dataloader = datamodule.get_dataloader(batch_size=2)
        
        self.assertEqual(dataloader.batch_size, 2)
    
    @patch(LOAD_DATASET_PATCH)
    def test_custom_dataset_name(self, mock_load_dataset):
        """Test using a custom dataset name."""
        mock_load_dataset.return_value = create_mock_dataset(1)
        
        custom_name = "custom/coco-dataset"
        datamodule = CoCoDataModule(dataset_name=custom_name, split="val")
        
        self.assertEqual(datamodule.dataset_name, custom_name)
        self.assertEqual(datamodule.split, "val")


class TestCreateCocoDatamodule(unittest.TestCase):
    """Test cases for the factory function."""
    
    @patch(LOAD_DATASET_PATCH)
    def test_factory_function(self, mock_load_dataset):
        """Test the factory function creates a valid datamodule."""
        mock_load_dataset.return_value = create_mock_dataset()
        
        datamodule = create_coco_datamodule(split="train", image_size=512)
        
        self.assertIsInstance(datamodule, CoCoDataModule)
        self.assertEqual(datamodule.split, "train")
        self.assertEqual(datamodule.image_size, 512)


class TestDataModuleConfig(unittest.TestCase):
    """Test cases for configuration dataclasses."""
    
    def test_datamodule_config_defaults(self):
        """Test DataModuleConfig default values."""
        config = DataModuleConfig(dataset_name="test/dataset")
        
        self.assertEqual(config.dataset_name, "test/dataset")
        self.assertEqual(config.split, "train")
        self.assertIsNone(config.cache_dir)
        self.assertFalse(config.streaming)
    
    def test_coco_config_defaults(self):
        """Test CoCoConfig default values."""
        config = CoCoConfig()
        
        self.assertEqual(config.dataset_name, "detection-datasets/coco")
        self.assertEqual(config.split, "train")
        self.assertFalse(config.return_masks)
        self.assertIsNone(config.image_size)


if __name__ == "__main__":
    unittest.main()
