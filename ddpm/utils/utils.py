import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import transforms, datasets
from utils.datasets import ColoredMNIST, CelebAHQ_dataset_multiconc, CUB_dataset_multiconc
from ast import literal_eval
import matplotlib.pyplot as plt
import textwrap
from PIL import Image, ImageDraw, ImageFont

def get_concept_index(model, c):
    if c==0:
        start=0
    else:
        start=sum(model.concept_bins[:c])
    end= sum(model.concept_bins[:c+1])

    return start,end

def sample_noise(num, dim, device=None) -> torch.Tensor:
    return torch.randn(num, dim, device=device)


def sample_code(num,model, return_list=False) -> torch.Tensor:
    cat_onehot = cont = bin = None
    output_code=None
    if(return_list):
        output_list = []

    for c in range(model.n_concepts):
        if(model.concept_type[c]=="cat"):
            cat_dim= model.concept_bins[c]
            cat = torch.randint(cat_dim, size=(num, 1), dtype=torch.long, device=model.device)
            cat_onehot = torch.zeros(num, cat_dim, dtype=torch.float, device=model.device)
            cat_onehot.scatter_(1, cat, 1)
            if(output_code==None):
                output_code=cat_onehot
            else:
                output_code=torch.cat((output_code,cat_onehot),1)
            if(return_list):
                output_list.append(cat_onehot)
        elif(model.concept_type[c]=="bin"):
            bin_dim= model.concepts_output[c]
            bin = (torch.rand(num, bin_dim, device=model.device) > .5).float()
            if(output_code==None):
                output_code=bin
            else:
                output_code=torch.cat((output_code,bin),1)
            if(return_list):
                output_list.append(bin.squeeze())
    if(return_list):
        return output_code,output_list
    else:
        return output_code

def sample_code_cmnist(num, model, concepts=[], return_list=False) -> torch.Tensor:
    cat_onehot = cont = bin = None
    output_code=None
    if(return_list):
        output_list = []

    for c in range(model.n_concepts):
        if(model.concept_type[c]=="cat"):
            cat_dim= model.concept_bins[c]
            try:
                cat = torch.tensor([concepts[c]] * num, dtype=torch.long, device=model.device).unsqueeze(1)
                assert cat.size() == torch.Size([num, 1])
            except:
                print(f'going into exception for categorical concept {c}')
                cat = torch.randint(cat_dim, size=(num, 1), dtype=torch.long, device=model.device)
            cat_onehot = torch.zeros(num, cat_dim, dtype=torch.float, device=model.device)
            cat_onehot.scatter_(1, cat, 1)
            if(output_code==None):
                output_code=cat_onehot
            else:
                output_code=torch.cat((output_code,cat_onehot),1)
            if(return_list):
                output_list.append(cat_onehot)
        elif(model.concept_type[c]=="bin"):
            bin_dim= model.concepts_output[c]
            try:
                bin_value = torch.tensor([concepts[c]] * num, dtype=torch.long, device=model.device).unsqueeze(1)
                bin = torch.zeros(num, bin_dim, dtype=torch.float, device=model.device)
                bin.scatter_(1, bin_value, 1)
            except:
                print(f'going into exception for binary concept {c}')
                bin = (torch.rand(num, bin_dim, device=model.device) > .5).float()
            if(output_code==None):
                output_code=bin
            else:
                output_code=torch.cat((output_code,bin),1)
            if(return_list):
                output_list.append(bin.squeeze())
    if(return_list):
        return output_code,output_list
    else:
        return output_code


def get_dataset(config,batch_size=None):
    if batch_size==None:
        batch_size=config["dataset"]["batch_size"]

    if(config["dataset"]["name"] =="color_mnist"):
        train_loader = torch.utils.data.DataLoader(
            ColoredMNIST(root='./data', env='train',
                     transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                         transforms.ToTensor(),
                         # transforms.Normalize(literal_eval(config["dataset"]["transforms_1"]), literal_eval(config["dataset"]["transforms_2"]))
                    ])),
            batch_size=batch_size,
            shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            ColoredMNIST(root='./data', env='test',
                     transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                         transforms.ToTensor(),
                         # transforms.Normalize(literal_eval(config["dataset"]["transforms_1"]), literal_eval(config["dataset"]["transforms_2"]))
                       ])),
            batch_size=config["dataset"]["test_batch_size"],
            shuffle=True,
        )

    elif config["dataset"]["name"] in ["celebahq", "celeba64", "cub", "cub64"]:
        # For CelebA-HQ, CelebA64, and CUB datasets
        img_size = config["dataset"]["img_size"]

        # Determine data path
        data_path = config["dataset"].get("data_path", f"./datasets/{config['dataset']['name'].upper()}")

        # Create transforms
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Check if we should use real labels
        use_real_labels = config["dataset"].get("use_real_labels", False)

        if use_real_labels and config["dataset"]["name"] == "celebahq":
            # Use CelebAHQ dataset with real concept labels
            import os
            from PIL import Image
            import numpy as np

            train_anno = config["dataset"].get("train_anno", os.path.join(data_path, "train_balanced.txt"))
            test_anno = config["dataset"].get("test_anno", os.path.join(data_path, "test_balanced.txt"))

            # Create a custom dataset class for the balanced annotation format
            class CelebAHQBalancedDataset(torch.utils.data.Dataset):
                def __init__(self, img_root, anno_path, transform=None):
                    self.img_root = img_root
                    self.transform = transform

                    with open(anno_path, 'r') as f:
                        lines = f.read().splitlines()

                    # First line is number of samples, second line is header
                    self.num_samples = int(lines[0])
                    self.concept_names = lines[1].split()

                    # Remaining lines are data
                    self.data = []
                    for line in lines[2:]:
                        parts = line.split()
                        img_name = parts[0]
                        labels = [int(x) for x in parts[1:]]
                        self.data.append((img_name, labels))

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    img_name, labels = self.data[idx]
                    # Remove leading zeros and .jpg, then add back .jpg
                    # e.g., "013881.jpg" -> "13881.jpg"
                    img_name_without_ext = img_name.replace('.jpg', '').lstrip('0') or '0'
                    img_name_fixed = img_name_without_ext + '.jpg'
                    # Images are in 'images' subdirectory
                    img_path = os.path.join(self.img_root, 'images', img_name_fixed)
                    image = Image.open(img_path).convert('RGB')

                    if self.transform:
                        image = self.transform(image)

                    return image, labels

            train_dataset = CelebAHQBalancedDataset(
                img_root=data_path,
                anno_path=train_anno,
                transform=train_transform
            )
            test_dataset = CelebAHQBalancedDataset(
                img_root=data_path,
                anno_path=test_anno,
                transform=test_transform
            )
            print(f"Loaded CelebA-HQ with real labels: {len(train_dataset)} train, {len(test_dataset)} test")

        elif use_real_labels and config["dataset"]["name"] in ["cub", "cub64"]:
            # Use CUB dataset with real concept labels
            import os
            image_path = config["dataset"].get("image_path", os.path.join(data_path, "images.txt"))
            anno_path = config["dataset"].get("anno_path", os.path.join(data_path, "attributes/image_attribute_labels.txt"))
            split_path = config["dataset"].get("split_path", os.path.join(data_path, "train_test_split.txt"))

            # Default CUB attributes
            set_of_classes = [219, 236, 55, 290, 152, 21, 245, 7, 36, 52]

            train_dataset = CUB_dataset_multiconc(
                img_root=os.path.join(data_path, "images"),
                image_path=image_path,
                anno_path=anno_path,
                split_path=split_path,
                set_of_classes=set_of_classes,
                transform=train_transform,
                split='train'
            )
            test_dataset = CUB_dataset_multiconc(
                img_root=os.path.join(data_path, "images"),
                image_path=image_path,
                anno_path=anno_path,
                split_path=split_path,
                set_of_classes=set_of_classes,
                transform=test_transform,
                split='test'
            )
            print(f"Loaded CUB with real labels: {len(train_dataset)} train, {len(test_dataset)} test")

        else:
            # Load dataset using ImageFolder (no labels)
            try:
                train_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
                test_dataset = datasets.ImageFolder(root=data_path, transform=test_transform)

                print(f"Loaded {len(train_dataset)} images from {data_path}")
            except Exception as e:
                print(f"Error loading dataset from {data_path}: {e}")
                print(f"Trying alternative structure...")
                # If ImageFolder fails, try loading images directly
                from glob import glob
                import os

                class SimpleImageDataset(torch.utils.data.Dataset):
                    def __init__(self, root, transform=None):
                        self.root = root
                        self.transform = transform
                        # Find all image files
                        self.image_files = []
                        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                            self.image_files.extend(glob(os.path.join(root, '**', ext), recursive=True))
                        print(f"Found {len(self.image_files)} images")

                    def __len__(self):
                        return len(self.image_files)

                    def __getitem__(self, idx):
                        from PIL import Image
                        img_path = self.image_files[idx]
                        img = Image.open(img_path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        return img, 0  # Return dummy label

                train_dataset = SimpleImageDataset(data_path, transform=train_transform)
                test_dataset = SimpleImageDataset(data_path, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["dataset"].get("num_workers", 4),
            drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["dataset"].get("test_batch_size", batch_size),
            shuffle=False,
            num_workers=config["dataset"].get("num_workers", 4),
        )

    return train_loader ,test_loader


# Modified from ChatGPT output
def create_image_grid(images, labels, probs, savefile, n_row=2, n_col=4, figsize=(10, 10), set_of_classes=None, dataset='color_mnist', textwidth=18):
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < len(images):
                # if dataset == 'color_mnist':
                #     image = images[idx].mul(255).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                # elif dataset == 'celebahq':
                #     image = images[idx].mul(127.5).add_(128).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                image = images[idx].mul(255).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(int)
                title = ''
                for cls_set_idx in range(len(labels)):
                    label = labels[cls_set_idx][idx]
                    prob = probs[cls_set_idx][idx]
                    curr_label_set = set_of_classes[cls_set_idx]
                    title += f'{curr_label_set[label]}, p={prob:.2f}'
                    if cls_set_idx != len(labels) - 1:
                        title += ' | '
                axes[i, j].imshow(image)
                title = '\n'.join(textwrap.wrap(title, width=textwidth))
                axes[i, j].set_title(title)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

# Modified from ChatGPT output
def save_image_grid_with_labels(image_tensor, class_indices, class_names, grid_size=(8, 8), file_name='image_grid.png', font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', return_images=False):
    """
    Save an image tensor as a grid of images with corresponding class names overlayed.
    
    Parameters:
    - image_tensor: A tensor of images (BxCxHxW)
    - class_indices: A list of class indices corresponding to each image
    - class_names: A list of all possible class names
    - grid_size: A tuple indicating the grid size (rows, cols)
    - file_name: The name of the file to save the image grid
    - font_path: Path to the font file for overlaying text
    """
    # Ensure the image tensor and class indices have the same length
    assert len(image_tensor) == len(class_indices), "The number of images must match the number of class indices"
    if image_tensor.shape[2] > 128:
        textsize = 16
    else:
        textsize = 6
 
    # Normalize the image tensor to the range [0, 1]
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    
    # Create a list to store images with class names overlayed
    images_with_text = []
    transform = transforms.ToPILImage()
    
    for img, idx in zip(image_tensor, class_indices):
        label = class_names[idx]
        img_pil = transform(img)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, textsize)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_position = (img_pil.width - text_size[0] - 5, img_pil.height - text_size[1] - 5)
        draw.text(text_position, label, font=font, fill="white")
        images_with_text.append(transforms.ToTensor()(img_pil))
    
    # Stack the images back into a tensor
    images_with_text_tensor = torch.stack(images_with_text)
    if return_images:
        return images_with_text_tensor
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(images_with_text_tensor, nrow=grid_size[1], padding=2, normalize=True)
    
    # Save the grid of images
    torchvision.utils.save_image(grid, file_name, normalize=True)

# Modified from ChatGPT output
def save_image_grid_with_otherconceptinfo(image_tensor, class_indices, class_names, text_list, grid_size=(8, 8), file_name='image_grid.png', font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
    """
    Save an image tensor as a grid of images with corresponding class names overlayed.
    
    Parameters:
    - image_tensor: A tensor of images (BxCxHxW)
    - class_indices: A list of class indices corresponding to each image
    - class_names: A list of all possible class names
    - grid_size: A tuple indicating the grid size (rows, cols)
    - file_name: The name of the file to save the image grid
    - font_path: Path to the font file for overlaying text
    """
    # Ensure the image tensor and class indices have the same length
    assert len(image_tensor) == len(class_indices), "The number of images must match the number of class indices"
    
    # Normalize the image tensor to the range [0, 1]
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    
    # Create a list to store images with class names overlayed
    images_with_text = []
    transform = transforms.ToPILImage()
    
    for img, idx, temp_text in zip(image_tensor, class_indices, text_list):
        label = class_names[idx] + '\n' + temp_text
        img_pil = transform(img)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, 16)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_position = (img_pil.width - text_size[0] - 5, img_pil.height - text_size[1] - 5)
        draw.text(text_position, label, font=font, fill="white")
        images_with_text.append(transforms.ToTensor()(img_pil))
    
    # Stack the images back into a tensor
    images_with_text_tensor = torch.stack(images_with_text)
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(images_with_text_tensor, nrow=grid_size[1], padding=2, normalize=True)
    
    # Save the grid of images
    torchvision.utils.save_image(grid, file_name, normalize=True)


def save_image_with_concept_probs_graph(image, probs, concepts, output_path, colors, image_size=(256, 256)):
    """
    Saves an image with a bar graph of concept prediction probabilities beside it.

    Parameters:
    - image: Tensor representing the image.
    - probs: List of concept prediction probabilities for the image.
    - concepts: List of concept names corresponding to the probabilities.
    - output_path: Path to save the combined image.
    - colors: List of hex color codes for the bars.
    - image_size: Size to which the image will be resized (default: (256, 256)).
    """
    # Convert the tensor image to PIL Image and resize
    transform = transforms.ToPILImage()
    img = transform(image).resize(image_size)
    img.save(output_path)

    # Create a bar chart for probabilities
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)  # Match the image size (256x256)
    # fontsize = 24
    # fontsize = 30
    fontsize = 40
    # fontsize = 60
    ax.barh(concepts, probs, color=colors, height=0.3)
    ax.set_xlim(0, 1)  # Probability range from 0 to 1
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)  # Vertical line at 0.0
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # X-ticks for 0, 0.5, and 1
    # ax.set_xticks([0, 0.25, 0.5, 0.75, 1])  # X-ticks for 0, 0.5, and 1

    # Set major ticks at 0, 0.5, and 1
    ax.set_xticks([0, 0.5, 1])
    # Set minor ticks at 0.25 and 0.75 (these will not have labels)
    ax.set_xticks([0.25, 0.75], minor=True)

    # Customize appearance: remove borders and horizontal lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color("gray")

    # Display only x-axis ticks without lines
    ax.tick_params(axis='x', direction='in', length=8, which='both', colors='black', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlabel("probability", fontsize=fontsize)

    # Save the bar chart as an image
    plt_path = output_path.replace(".png", "_bar.png")
    plt.savefig(plt_path, bbox_inches="tight", facecolor='white')
    plt.close(fig)

    # # Open the bar chart image and combine it with the original image
    # bar_img = Image.open(plt_path).resize(image_size)
    # combined_width = img.width + bar_img.width
    # combined_height = max(img.height, bar_img.height)
    # combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    # combined_img.paste(img, (0, 0))
    # combined_img.paste(bar_img, (img.width, 0))

    # # Save the final combined image
    # combined_img.save(output_path)

    # Open the high-res bar chart image and resize it to 128x256
    # bar_img = Image.open(plt_path).resize((256, 128), Image.BICUBIC)
    # # Create a new image with the original image on top and the resized bar graph at the bottom
    # combined_width = max(img.width, bar_img.width)
    # combined_height = img.height + bar_img.height
    # combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    # combined_img.paste(img, (0, 0))
    # combined_img.paste(bar_img, (0, img.height))

    # # Save the final combined image
    # combined_img.save(output_path)
