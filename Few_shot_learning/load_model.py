from modules.episodic_prototypical import PrototypicalNetworks
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from fewshot_sampler import FewShotSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from wrap_few_shot_dataset import WrapFewShotDataset

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images, support_labels, query_images)
            .detach()
            .data,
            1,
        )[1]
        == query_labels
    ).sum().item(), len(query_labels)


def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()

    with torch.no_grad():
        for (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(data_loader, total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )

if __name__ == "__main__":

    image_size = 28

    # Setup path to data folder
    data_path = Path("data")
    image_path = data_path / "EuroSAT"

    # Check if image folder exists
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory")
        exit()

    # Setup train and testing paths
    test_dir = image_path / "Test"

    test_transform = transforms.Compose([
        transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    test_set = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform,
    )

    test_set = WrapFewShotDataset(test_set)

    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetworks(convolutional_network)
    model.load_state_dict(torch.load("models/fewshot_merced_proto.pth",map_location=torch.device('cpu')))


    N_WAY = 3  # Number of classes in a task
    N_SHOT = 1  # Number of images per class in the support set
    N_QUERY = 10  # Number of images per class in the query set
    N_EVALUATION_TASKS = 100

    test_sampler = FewShotSampler (
        test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    model.eval()
    evaluate(test_loader)