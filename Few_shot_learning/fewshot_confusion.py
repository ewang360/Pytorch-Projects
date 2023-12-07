from modules.episodic_prototypical import PrototypicalNetworks
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50
from fewshot_sampler import FewShotSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from wrap_few_shot_dataset import WrapFewShotDataset

from sklearn.metrics import confusion_matrix
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def get_labels(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    class_ids: torch.Tensor
):
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """

    predicted_tensor = torch.max(
            model(support_images, support_labels, query_images)
            .detach()
            .data,
            1,
        )[1].tolist()
    
    pred_labels = []
    class_labels = []

    for i, label in enumerate(query_labels):
        pred_labels.append(class_ids[predicted_tensor[i]])
        class_labels.append(class_ids[label])

    return pred_labels, class_labels


def get_data(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    pred = []
    true = []

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

            correct, total = get_labels(
                support_images, support_labels, query_images, query_labels, class_ids
            )

            pred.extend(correct)
            true.extend(total)

    return pred, true

if __name__ == "__main__":

    image_size = 28

    # Setup path to data folder
    data_path = Path("data")
    image_path = data_path / "UCMerced-Fewshot"

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


    N_WAY = 5  # Number of classes in a task
    N_SHOT = 5  # Number of images per class in the support set
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
    pred, true = get_data(test_loader)

    # generate confusion matrix
    classes = ('buildings', 'chaparral', 'denseresidential', 'intersection', 'mediumresidential',
        'mobilehomepark', 'sparseresidential', 'storagetanks', 'tenniscourt')
    # classes = ('AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake')
    
    cf_matrix = confusion_matrix(true,pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    
    plt.figure(figsize = (12,7), dpi=2000)
    sn.heatmap(df_cm, annot=True)
    plt.title("UCMerced Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig('confusion_matrices/UCMerced_test_5shot.png')