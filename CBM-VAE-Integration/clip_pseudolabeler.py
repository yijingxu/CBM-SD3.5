"""
CLIP Pseudo-Labeler for Concept Validation

This module provides CLIP-based pseudo-labeling for concept validation during training.
Based on the reference implementation from posthoc-generative-cbm.
"""

import torch
import torch.nn as nn
import clip


class CLIPPseudoLabeler:
    """
    CLIP-based pseudo-labeler for binary concepts.

    Uses CLIP to determine if concepts (smiling, glasses) are present in generated images.
    """

    def __init__(self, device='cuda'):
        """
        Initialize CLIP model and text prompts.

        Args:
            device: Device to run CLIP on
        """
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        # Concept descriptions [negative, positive]
        self.concept_texts = [
            ["a photo of a person not smiling", "a photo of a person smiling"],
            ["a photo of a person without glasses", "a photo of a person with glasses"]
        ]

        # Precompute text features for efficiency
        self.text_features_list = []
        with torch.no_grad():
            for concept_texts in self.concept_texts:
                text_tokens = clip.tokenize(concept_texts).to(device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_features_list.append(text_features)

        print(f"Initialized CLIP PseudoLabeler with {len(self.concept_texts)} concepts")

    def preprocess_images(self, images):
        """
        Preprocess images for CLIP.

        Args:
            images: Tensor of shape [B, C, H, W] in range [0, 1]

        Returns:
            Preprocessed images for CLIP
        """
        # CLIP expects 224x224 images normalized with specific mean/std
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711])
        ])

        return transform(images)

    def get_pseudo_labels(self, images, return_prob=True):
        """
        Get pseudo-labels from CLIP for generated images.

        Args:
            images: Generated images [B, C, H, W] in range [0, 1]
            return_prob: If True, return probabilities; else return class indices

        Returns:
            If return_prob=True: (probs_list, labels_list)
                probs_list: List of [B] tensors with probabilities for positive class
                labels_list: List of [B] tensors with class indices (0 or 1)
            If return_prob=False: labels_list only
        """
        with torch.no_grad():
            # Preprocess images for CLIP
            images_preprocessed = self.preprocess_images(images)

            # Encode images
            image_features = self.model.encode_image(images_preprocessed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            probs_list = []
            labels_list = []

            for text_features in self.text_features_list:
                # Compute similarity
                similarity = image_features @ text_features.T  # [B, 2]
                probs = similarity.softmax(dim=-1)  # [B, 2]

                # Get predicted class (0=negative, 1=positive)
                _, predicted_class = torch.max(probs, dim=1)

                probs_list.append(probs[:, 1])  # Probability of positive class
                labels_list.append(predicted_class)

            if return_prob:
                return probs_list, labels_list
            else:
                return labels_list

    def get_soft_pseudo_labels(self, images):
        """
        Get soft logits for intervention training.

        Args:
            images: Generated images [B, C, H, W] in range [0, 1]

        Returns:
            List of [B, 2] logit tensors for each concept
        """
        with torch.no_grad():
            # Preprocess images for CLIP
            images_preprocessed = self.preprocess_images(images)

            # Encode images
            image_features = self.model.encode_image(images_preprocessed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits_list = []

            for text_features in self.text_features_list:
                # Compute similarity (acts as logits)
                similarity = image_features @ text_features.T  # [B, 2]
                logits_list.append(similarity * 100)  # Scale for numerical stability

            return logits_list

    def validate_intervention(self, original_image, intervened_image, concept_idx, target_value):
        """
        Validate that an intervention actually changed the image as expected.

        Args:
            original_image: Original generated image [C, H, W]
            intervened_image: Image after intervention [C, H, W]
            concept_idx: Which concept was intervened (0=smiling, 1=glasses)
            target_value: Target value for the concept (0 or 1)

        Returns:
            success: True if CLIP recognizes the intervention
            confidence: Probability of the target concept
        """
        # Add batch dimension
        images = torch.stack([original_image, intervened_image])

        probs_list, _ = self.get_pseudo_labels(images, return_prob=True)

        # Check the intervened concept
        intervened_prob = probs_list[concept_idx][1]  # Probability for intervened image

        # Success if probability matches target (>0.5 for positive, <0.5 for negative)
        if target_value == 1:
            success = intervened_prob > 0.5
        else:
            success = intervened_prob < 0.5

        return success, intervened_prob.item()


if __name__ == "__main__":
    # Test the pseudo-labeler
    print("Testing CLIP PseudoLabeler...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labeler = CLIPPseudoLabeler(device=device)

    # Create dummy images
    batch_size = 4
    dummy_images = torch.rand(batch_size, 3, 512, 512, device=device)

    # Get pseudo-labels
    probs_list, labels_list = labeler.get_pseudo_labels(dummy_images, return_prob=True)

    print(f"\nSmiling probabilities: {probs_list[0]}")
    print(f"Smiling labels: {labels_list[0]}")
    print(f"Glasses probabilities: {probs_list[1]}")
    print(f"Glasses labels: {labels_list[1]}")

    # Get soft labels
    logits_list = labeler.get_soft_pseudo_labels(dummy_images)
    print(f"\nSmiling logits shape: {logits_list[0].shape}")
    print(f"Glasses logits shape: {logits_list[1].shape}")

    print("\nCLIP PseudoLabeler test completed successfully!")
