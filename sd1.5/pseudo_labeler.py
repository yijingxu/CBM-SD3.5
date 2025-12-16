"""
Pseudo-labelers for concept prediction.

Provides different sources for concept labels:
- CLIP zero-shot: Uses CLIP to predict concepts from text descriptions
- Supervised: Uses pretrained classifiers on CelebA attributes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import clip
from torchvision import transforms, models
import os


class CLIP_PseudoLabeler:
    """
    CLIP-based zero-shot concept labeler.
    
    Uses CLIP to predict concepts by comparing image embeddings
    with text embeddings of concept descriptions.
    """
    
    def __init__(
        self,
        concept_classes: List[List[str]],
        device: str = "cuda",
        clip_model: str = "ViT-B/32",
    ):
        """
        Args:
            concept_classes: List of concepts, each concept is a list of class names
                e.g., [['NOT Smiling', 'Smiling'], ['Female', 'Male']]
            device: Device to use
            clip_model: CLIP model variant
        """
        self.device = device
        self.concept_classes = concept_classes
        self.n_concepts = len(concept_classes)
        
        # Load CLIP model
        print(f"Loading CLIP model: {clip_model}")
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model.eval()
        
        # Precompute text embeddings for all concepts
        self._compute_text_embeddings()
        
        # Image preprocessing for CLIP (different from training transforms)
        self.clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
        
    def _compute_text_embeddings(self):
        """Precompute text embeddings for all concept classes."""
        self.text_embeddings = []
        
        with torch.no_grad():
            for concept_idx, classes in enumerate(self.concept_classes):
                # Create text prompts
                prompts = [f"a photo of a person who is {c.lower()}" for c in classes]
                tokens = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_embeddings.append(text_features)
                
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP.
        
        Args:
            images: Images in range [0, 1] or [-1, 1], shape [B, 3, H, W]
            
        Returns:
            Preprocessed images for CLIP
        """
        # Ensure images are in [0, 1] range
        if images.min() < 0:
            images = (images + 1) / 2
        
        # Apply CLIP preprocessing
        images = self.clip_transform(images)
        return images
    
    @torch.no_grad()
    def get_pseudo_labels(
        self,
        images: torch.Tensor,
        return_prob: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get pseudo-labels for images.
        
        Args:
            images: Images tensor [B, 3, H, W]
            return_prob: Whether to return probabilities
            
        Returns:
            If return_prob:
                (probs_list, labels_list): Lists of probabilities and labels per concept
            Else:
                labels_list: List of label tensors per concept
        """
        # Preprocess for CLIP
        images = self._preprocess_images(images)
        
        # Get image features
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        labels_list = []
        probs_list = []
        
        for concept_idx, text_features in enumerate(self.text_embeddings):
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get predictions
            probs, labels = similarity.max(dim=-1)
            
            labels_list.append(labels)
            probs_list.append(probs)
        
        if return_prob:
            return probs_list, labels_list
        return labels_list
    
    @torch.no_grad()
    def get_soft_pseudo_labels(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Get soft pseudo-labels (logits) for images.
        
        Args:
            images: Images tensor [B, 3, H, W]
            
        Returns:
            List of logit tensors per concept, each [B, num_classes]
        """
        # Preprocess for CLIP
        images = self._preprocess_images(images)
        
        # Get image features
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits_list = []
        
        for text_features in self.text_embeddings:
            # Compute similarity as logits
            logits = 100.0 * image_features @ text_features.T
            logits_list.append(logits.float())
        
        return logits_list


class SupervisedPseudoLabeler:
    """
    Supervised classifier-based pseudo-labeler.
    
    Uses pretrained classifiers for each concept attribute.
    """
    
    def __init__(
        self,
        concept_classes: List[List[str]],
        device: str = "cuda",
        checkpoint_dir: str = "models/classifiers/",
        model_type: str = "resnet18",
    ):
        """
        Args:
            concept_classes: List of concepts
            device: Device to use
            checkpoint_dir: Directory containing classifier checkpoints
            model_type: Base model architecture
        """
        self.device = device
        self.concept_classes = concept_classes
        self.n_concepts = len(concept_classes)
        self.checkpoint_dir = checkpoint_dir
        
        # Load classifiers for each concept
        self.classifiers = nn.ModuleList()
        for i, classes in enumerate(concept_classes):
            n_classes = len(classes)
            classifier = self._create_classifier(model_type, n_classes)
            
            # Try to load checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"concept_{i}.pt")
            if os.path.exists(ckpt_path):
                classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Loaded classifier for concept {i}: {classes}")
            else:
                print(f"Warning: No checkpoint found for concept {i} at {ckpt_path}")
            
            classifier.eval()
            self.classifiers.append(classifier)
        
        self.classifiers.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
    def _create_classifier(self, model_type: str, n_classes: int) -> nn.Module:
        """Create a classifier model."""
        if model_type == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif model_type == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return model
    
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for classifier."""
        # Ensure images are in [0, 1] range
        if images.min() < 0:
            images = (images + 1) / 2
        return self.transform(images)
    
    @torch.no_grad()
    def get_pseudo_labels(
        self,
        images: torch.Tensor,
        return_prob: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get pseudo-labels from supervised classifiers."""
        images = self._preprocess_images(images)
        
        labels_list = []
        probs_list = []
        
        for classifier in self.classifiers:
            logits = classifier(images)
            probs = F.softmax(logits, dim=-1)
            max_probs, labels = probs.max(dim=-1)
            
            labels_list.append(labels)
            probs_list.append(max_probs)
        
        if return_prob:
            return probs_list, labels_list
        return labels_list
    
    @torch.no_grad()
    def get_soft_pseudo_labels(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Get soft pseudo-labels (logits) from supervised classifiers."""
        images = self._preprocess_images(images)
        
        logits_list = []
        for classifier in self.classifiers:
            logits = classifier(images)
            logits_list.append(logits)
        
        return logits_list


class CelebAAttributeClassifier(nn.Module):
    """
    Multi-attribute classifier for CelebA.
    
    A single model that predicts multiple binary attributes simultaneously.
    """
    
    def __init__(
        self,
        n_attributes: int,
        model_type: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        
        if model_type == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(512, n_attributes)
        elif model_type == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(2048, n_attributes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.n_attributes = n_attributes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Images [B, 3, H, W]
            
        Returns:
            logits: Attribute logits [B, n_attributes]
        """
        return self.backbone(x)


class CelebAMultiAttributeLabeler:
    """
    Labeler using a multi-attribute CelebA classifier.
    """
    
    def __init__(
        self,
        concept_names: List[str],
        device: str = "cuda",
        checkpoint_path: str = None,
        model_type: str = "resnet18",
    ):
        """
        Args:
            concept_names: List of attribute names to predict
            device: Device to use
            checkpoint_path: Path to classifier checkpoint
            model_type: Base model architecture
        """
        self.device = device
        self.concept_names = concept_names
        self.n_concepts = len(concept_names)
        
        # Create classifier
        self.classifier = CelebAAttributeClassifier(
            n_attributes=self.n_concepts,
            model_type=model_type,
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.classifier.load_state_dict(
                torch.load(checkpoint_path, map_location=device)
            )
            print(f"Loaded multi-attribute classifier from {checkpoint_path}")
        
        self.classifier.to(device)
        self.classifier.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for classifier."""
        if images.min() < 0:
            images = (images + 1) / 2
        return self.transform(images)
    
    @torch.no_grad()
    def get_pseudo_labels(
        self,
        images: torch.Tensor,
        return_prob: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get pseudo-labels from multi-attribute classifier."""
        images = self._preprocess_images(images)
        logits = self.classifier(images)
        
        labels_list = []
        probs_list = []
        
        for i in range(self.n_concepts):
            # Binary classification: logit > 0 means positive class
            attr_logits = logits[:, i]
            probs = torch.sigmoid(attr_logits)
            labels = (probs > 0.5).long()
            
            labels_list.append(labels)
            probs_list.append(probs)
        
        if return_prob:
            return probs_list, labels_list
        return labels_list
    
    @torch.no_grad()
    def get_soft_pseudo_labels(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Get soft pseudo-labels as 2-class logits for each attribute.
        
        Returns logits in format compatible with cross-entropy loss.
        """
        images = self._preprocess_images(images)
        logits = self.classifier(images)
        
        logits_list = []
        for i in range(self.n_concepts):
            # Convert single logit to 2-class format: [negative_logit, positive_logit]
            attr_logit = logits[:, i:i+1]  # [B, 1]
            two_class_logits = torch.cat([-attr_logit, attr_logit], dim=1)  # [B, 2]
            logits_list.append(two_class_logits)
        
        return logits_list
