import torch
from torch import Tensor
from torch.utils.data import Sampler
from few_shot_dataset import FewShotDataset

import random
from typing import Dict, List, Tuple, Union

GENERIC_TYPING_ERROR_MESSAGE = (
    "Check out the output's type of your dataset's __getitem__() method."
    "It must be a Tuple[Tensor, int] or Tuple[Tensor, 0-dim Tensor]."
)

class FewShotSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: FewShotDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. 
            Must have implemented get_labels() from FewShotDataset.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label: Dict[int, List[int]] = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        """
        Sample n_way labels uniformly at random,
        and then sample n_shot + n_query items for each label, also uniformly at random.
        Yields:
            a list of indices of length (n_way * (n_shot + n_query))
        """
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    for label in random.sample(
                        sorted(self.items_per_label.keys()), self.n_way
                    )
                ]
            ).tolist()

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images of shape (n_way * n_shot, n_channels, height, width),
                - their labels of shape (n_way * n_shot),
                - query images of shape (n_way * n_query, n_channels, height, width)
                - their labels of shape (n_way * n_query),
                - the dataset class ids of the class sampled in the episode
        """
        input_data_with_int_labels = self._cast_input_data_to_tensor_int_tuple(
            input_data
        )
        true_class_ids = list({x[1] for x in input_data_with_int_labels})
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data_with_int_labels])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data_with_int_labels]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()
        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )
    
    @staticmethod
    def _cast_input_data_to_tensor_int_tuple(
        input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> List[Tuple[Tensor, int]]:
        """
        Check the type of the input for the episodic_collate_fn method, and cast it to the right type if possible.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            the input data with the labels cast to int
        Raises:
            TypeError : Wrong type of input images or labels
            ValueError: Input label is not a 0-dim tensor
        """
        for image, label in input_data:
            if not isinstance(image, Tensor):
                raise TypeError(
                    f"Illegal type of input instance: {type(image)}. "
                    + GENERIC_TYPING_ERROR_MESSAGE
                )
            if not isinstance(label, int):
                if label.dtype not in {
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                }:
                    raise TypeError(
                        f"Illegal dtype of input label tensor: {label.dtype}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )
                if label.ndim != 0:
                    raise ValueError(
                        f"Illegal shape for input label tensor: {label.shape}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )

        return [(image, int(label)) for (image, label) in input_data]