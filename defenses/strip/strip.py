"""
STRIP Defense for Backdoor Attacks

Adapted from the Input-Aware Dynamic Backdoor Attack paper implementation.

The defense works at inference time by:
1. For each input, superimpose it with N random clean images
2. Feed each superimposed image through the model
3. If input is clean: predictions will vary (high entropy)
4. If input is trojaned: predictions will be consistent (low entropy)
   -> The backdoor trigger dominates regardless of the blending
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# cv2 is optional - used by paper's superimposition, falls back to numpy
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.classifier import PreActResNet18
from models.generator import TriggerGenerator
from models.mask_generator import MaskGenerator
from util import inject_trigger


class Normalize:
    """Channel-wise normalization for images."""
    
    def __init__(self, n_channels, expected_values, variance):
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    """Channel-wise denormalization for images."""
    
    def __init__(self, n_channels, expected_values, variance):
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class STRIP:
    """
    STRIP detector following the paper's implementation.
    
    Uses cv2.addWeighted for image superimposition and computes entropy
    based on sigmoid probabilities (as in the original paper code).
    """
    
    def __init__(self, n_sample, n_channels, device, dataset="cifar10"):
        """
        Args:
            n_sample: Number of superimposition samples per input
            n_channels: Number of image channels (1 for MNIST, 3 for CIFAR10)
            device: torch device
            dataset: Dataset name for normalization parameters
        """
        self.n_sample = n_sample
        self.n_channels = n_channels
        self.device = device
        
        # Set up normalization based on dataset
        if dataset == "cifar10":
            self.normalizer = Normalize(n_channels, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            self.denormalizer = Denormalize(n_channels, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif dataset == "mnist":
            self.normalizer = Normalize(n_channels, [0.5], [0.5])
            self.denormalizer = Denormalize(n_channels, [0.5], [0.5])
        elif dataset == "gtsrb":
            self.normalizer = None
            self.denormalizer = None
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def _superimpose(self, background, overlay):
        """Superimpose two images using cv2.addWeighted or numpy fallback."""
        if HAS_CV2:
            output = cv2.addWeighted(background, 1, overlay, 1, 0)
        else:
            # Numpy fallback: simple addition with clipping
            output = np.clip(background.astype(np.int32) + overlay.astype(np.int32), 0, 255).astype(np.uint8)
        
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output
    
    def normalize(self, x):
        """Apply normalization transform."""
        if self.normalizer:
            # Convert HWC numpy to tensor, normalize, return tensor
            x_tensor = torch.from_numpy(x).float()
            x_tensor = self.normalizer(x_tensor)
            return x_tensor
        return torch.from_numpy(x).float()
    
    def denormalize(self, x):
        """Apply denormalization transform."""
        if self.denormalizer:
            return self.denormalizer(x)
        return x
    
    def _get_entropy(self, background, dataset, classifier):
        """
        Compute average entropy for a single input after superimposition.
        
        Args:
            background: Single image as numpy array (H, W, C), uint8
            dataset: Dataset to sample overlay images from
            classifier: The classifier model
            
        Returns:
            Average entropy across all superimposed samples
        """
        x1_add = []
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        
        for index in range(self.n_sample):
            # Get overlay image from dataset (returns tensor)
            overlay_tensor, _ = dataset[index_overlay[index]]
            
            # Convert tensor (C, H, W) to numpy (H, W, C) uint8
            overlay = overlay_tensor.numpy()
            overlay = (overlay * 255).astype(np.uint8)
            overlay = overlay.transpose((1, 2, 0))
            
            # Superimpose
            add_image = self._superimpose(background, overlay)
            
            # Normalize and add to batch
            add_image_normalized = self.normalize(add_image)
            # Convert from HWC to CHW for model
            add_image_normalized = add_image_normalized.permute(2, 0, 1)
            x1_add.append(add_image_normalized)
        
        # Stack and predict
        batch = torch.stack(x1_add).to(self.device)
        
        with torch.no_grad():
            py1_add = classifier(batch)
            py1_add = torch.sigmoid(py1_add).cpu().numpy()
        
        # Compute entropy: H = -sum(p * log2(p))
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add + 1e-10))
        
        return entropy_sum / self.n_sample
    
    def __call__(self, background, dataset, classifier):
        """
        Compute entropy for a background image.
        
        Args:
            background: Image as numpy array (H, W, C), uint8
            dataset: Dataset to sample overlay images from
            classifier: The classifier model
            
        Returns:
            Average entropy value
        """
        return self._get_entropy(background, dataset, classifier)


def strip_evaluate(checkpoint_path, data_root, dataset="cifar10", n_sample=100,
                   n_test=100, test_rounds=1, detection_boundary=0.2,
                   results_dir="./results", mode="attack"):
    """
    Run STRIP evaluation following the paper's approach.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Root directory for datasets
        dataset: Dataset name ("cifar10", "mnist", "gtsrb")
        n_sample: Number of superimposition samples per input
        n_test: Number of test samples per round
        test_rounds: Number of test rounds
        detection_boundary: Entropy threshold for detection
        results_dir: Directory to save results
        mode: "attack" to test trojaned samples, "clean" for clean only
        
    Returns:
        tuple: (list_entropy_trojan, list_entropy_benign)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Set up dataset parameters
    if dataset == "mnist":
        input_height, input_width, input_channel = 28, 28, 1
        num_classes = 10
    elif dataset == "cifar10":
        input_height, input_width, input_channel = 32, 32, 3
        num_classes = 10
    elif dataset == "gtsrb":
        input_height, input_width, input_channel = 32, 32, 3
        num_classes = 43
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset == "cifar10":
        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])
        testset = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset} not implemented")
    
    test_loader = DataLoader(testset, batch_size=n_test, shuffle=True,
                             num_workers=4, pin_memory=True)
    
    # Initialize classifier
    classifier = PreActResNet18(num_classes=num_classes, in_channels=input_channel).to(device)
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(state_dict["classifier"])
    
    # Initialize generators if testing attack mode
    generator = None
    mask_gen = None
    if mode == "attack":
        generator = TriggerGenerator(inChannels=input_channel, outChannels=input_channel).to(device)
        mask_gen = MaskGenerator(inChannels=input_channel).to(device)
        
        generator.load_state_dict(state_dict["generator"])
        mask_gen.load_state_dict(state_dict["mask_generator"])
        
        generator.eval()
        mask_gen.eval()
        for param in generator.parameters():
            param.requires_grad = False
        for param in mask_gen.parameters():
            param.requires_grad = False
    
    # Initialize STRIP detector
    strip_detector = STRIP(n_sample, input_channel, device, dataset)
    
    # Collect entropies over multiple rounds
    lists_entropy_trojan = []
    lists_entropy_benign = []
    
    for test_round in range(test_rounds):
        print(f"\n=== Test Round {test_round + 1}/{test_rounds} ===")
        
        list_entropy_trojan = []
        list_entropy_benign = []
        
        if mode == "attack":
            # Testing with backdoored data
            print("Testing with backdoored data...")
            inputs, targets = next(iter(test_loader))
            inputs = inputs.to(device)
            
            # Generate trojaned images
            with torch.no_grad():
                patterns = generator(inputs)
                masks = mask_gen.threshold(mask_gen(inputs))
                bd_inputs = inject_trigger(inputs, patterns, masks)
            
            # Convert to numpy uint8 format for STRIP
            bd_inputs_np = bd_inputs.detach().cpu().numpy()
            bd_inputs_np = (np.clip(bd_inputs_np, 0, 1) * 255).astype(np.uint8)
            bd_inputs_np = bd_inputs_np.transpose((0, 2, 3, 1))  # NCHW -> NHWC
            
            for index in tqdm(range(n_test), desc="Trojaned samples"):
                background = bd_inputs_np[index]
                entropy = strip_detector(background, testset, classifier)
                list_entropy_trojan.append(entropy)
        
        # Testing with clean data
        print("Testing with clean data...")
        for index in tqdm(range(n_test), desc="Clean samples"):
            # Get image and convert to numpy uint8
            img_tensor, _ = testset[index]
            background = img_tensor.numpy()
            background = (background * 255).astype(np.uint8)
            background = background.transpose((1, 2, 0))  # CHW -> HWC
            
            entropy = strip_detector(background, testset, classifier)
            list_entropy_benign.append(entropy)
        
        lists_entropy_trojan.extend(list_entropy_trojan)
        lists_entropy_benign.extend(list_entropy_benign)
    
    # Save results
    result_dir = os.path.join(results_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"strip_{mode}_output.txt")
    
    with open(result_path, "w") as f:
        if lists_entropy_trojan:
            f.write(" ".join(map(str, lists_entropy_trojan)))
            f.write("\n")
        f.write(" ".join(map(str, lists_entropy_benign)))
    
    print(f"\nResults saved to: {result_path}")
    
    # Determine if backdoored
    all_entropies = lists_entropy_trojan + lists_entropy_benign
    if all_entropies:
        min_entropy = min(all_entropies)
        print(f"\nMin entropy: {min_entropy:.4f}, Detection boundary: {detection_boundary}")
        
        if min_entropy < detection_boundary:
            print("Result: Detected as BACKDOORED model")
        else:
            print("Result: Model appears CLEAN")
    
    return lists_entropy_trojan, lists_entropy_benign


# ============================================================================
# Alternative PyTorch-native implementation (more efficient for batched ops)
# ============================================================================

class STRIPDetectorPyTorch:
    """
    Alternative STRIP detector using pure PyTorch operations.
    More efficient for GPU batch processing.
    """
    
    def __init__(self, model, clean_dataset, device, 
                 n_perturbations=100, alpha=0.5):
        self.model = model
        self.device = device
        self.n_perturbations = n_perturbations
        self.alpha = alpha
        
        # Pre-load clean images for blending
        self.clean_images = []
        num_samples = min(1000, len(clean_dataset))
        indices = torch.randperm(len(clean_dataset))[:num_samples]
        
        for idx in indices:
            img, _ = clean_dataset[idx]
            self.clean_images.append(img)
        
        self.clean_images = torch.stack(self.clean_images).to(device)
        self.entropy_threshold = None
    
    def _compute_entropy(self, inputs):
        """Compute prediction entropy using PyTorch operations."""
        batch_size = inputs.size(0)
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.n_perturbations):
                indices = torch.randint(0, len(self.clean_images), (batch_size,))
                clean_samples = self.clean_images[indices]
                
                # Alpha blending
                perturbed = self.alpha * inputs + (1 - self.alpha) * clean_samples
                
                logits = self.model(perturbed)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        all_probs = torch.stack(all_probs, dim=0)
        avg_probs = all_probs.mean(dim=0)
        entropies = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1)
        
        return entropies
    
    def calibrate(self, clean_loader, percentile=1.0):
        """Calibrate threshold on clean data."""
        print("Calibrating STRIP detector on clean data...")
        
        all_entropies = []
        for images, _ in tqdm(clean_loader, desc="Computing entropies"):
            images = images.to(self.device)
            entropies = self._compute_entropy(images)
            all_entropies.append(entropies.cpu())
        
        all_entropies = torch.cat(all_entropies).numpy()
        self.entropy_threshold = np.percentile(all_entropies, percentile)
        
        print(f"  Threshold ({percentile}th percentile): {self.entropy_threshold:.4f}")
        return self.entropy_threshold
    
    def detect(self, inputs):
        """Detect trojaned inputs."""
        inputs = inputs.to(self.device)
        entropies = self._compute_entropy(inputs)
        is_trojaned = entropies < self.entropy_threshold
        return is_trojaned.cpu(), entropies.cpu()


def main():
    parser = argparse.ArgumentParser(description="STRIP Defense (Paper Implementation)")
    
    # Paths
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    
    # STRIP parameters
    parser.add_argument("--n_sample", type=int, default=100,
                        help="Number of superimposition samples per input")
    parser.add_argument("--n_test", type=int, default=100,
                        help="Number of test samples")
    parser.add_argument("--test_rounds", type=int, default=1,
                        help="Number of test rounds")
    parser.add_argument("--detection_boundary", type=float, default=0.2,
                        help="Entropy threshold for backdoor detection")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "mnist", "gtsrb"])
    
    # Mode
    parser.add_argument("--mode", type=str, default="attack",
                        choices=["attack", "clean"],
                        help="'attack' to test trojaned samples, 'clean' for clean only")
    
    args = parser.parse_args()
    
    # Run evaluation
    strip_evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        dataset=args.dataset,
        n_sample=args.n_sample,
        n_test=args.n_test,
        test_rounds=args.test_rounds,
        detection_boundary=args.detection_boundary,
        results_dir=args.results_dir,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
