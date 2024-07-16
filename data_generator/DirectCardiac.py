import pathlib
import logging
from torch.utils.data import Dataset
from direct.types import PathOrString
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
logger = logging.getLogger(__name__)

"""
This code should be copied in the 'direct' package under  direct/data/datasets.py
"""


class MyNewDataset(Dataset):
    """
    Information about the Dataset.
    """

    def __init__(
        self,
        root: pathlib.Path,
        transform: Optional[Callable] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        text_description: Optional[str] = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        root : pathlib.Path
            Root directory to saved data.
        transform : Optional[Callable]
            Callable function that transforms the loaded data.
        filenames_filter : List
            List of filenames to include in the dataset.
        text_description : str
            Description of dataset, can be useful for logging.
        ...
        ...
        """
        super().__init__()

        self.logger = logging.getLogger(type(self).__name__)
        self.root = root
        self.transform = transform
        if filenames_filter:
            self.logger.info(f"Attempting to load {len(filenames_filter)} filenames from list.")
            filenames = filenames_filter
        else:
            self.logger.info(f"Parsing directory {self.root} for <data_type> files.")
        filenames = list(self.root.glob("*.<data_type>"))
        self.filenames_filter = filenames_filter
        self.text_description = text_description

    def get_dataset_len(self):
        # Now sure why we have two methods for this...
        return len(self.filenames_filter)

    def __len__(self):
        return self.get_dataset_len()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...
        sample = ...
        ...
        if self.transform:
            sample = self.transform(sample)
        return sample