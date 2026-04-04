"""
Neural Cleanse Defense for Backdoor Attacks

Adapted from the Input-Aware Dynamic Backdoor Attack paper implementation.

The defense works by:
1. For each class, reverse-engineer a minimal trigger that causes misclassification
2. Use an anomaly index (MAD-based) to detect which class has a backdoor
3. The class with anomaly index > threshold is flagged as the target class
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.classifier import PreActResNet18
from models.generator import TriggerGenerator
from models.mask_generator import MaskGenerator
from util import inject_trigger


class Recorder:
    """Records best mask/pattern found during optimization."""
    
    def __init__(self):
        self.mask_best = None
        self.pattern_best = None
        self.loss_best = float('inf')
        self.acc_best = 0.0
    
    def update(self, mask, pattern, loss, acc):
        if loss < self.loss_best:
            self.mask_best = mask.detach().clone()
            self.pattern_best = pattern.detach().clone()
            self.loss_best = loss
            self.acc_best = acc


def train(opt, init_mask, init_pattern, model, dataloader, device):
    """
    Reverse-engineer trigger for a single target label.
    
    This follows the paper's training approach.
    
    Args:
        opt: Options object with target_label, input dimensions, etc.
        init_mask: Initial mask array (1, H, W)
        init_pattern: Initial pattern array (C, H, W)
        model: The classifier model
        dataloader: Data loader for clean images
        device: torch device
        
    Returns:
        recorder: Recorder with best mask/pattern
        opt: Updated options
    """
    # Initialize learnable mask and pattern
    mask = torch.from_numpy(init_mask).float().to(device)
    mask.requires_grad = True
    
    pattern = torch.from_numpy(init_pattern).float().to(device)
    pattern.requires_grad = True
    
    # Optimizer
    optimizer = optim.Adam([mask, pattern], lr=opt.lr, betas=(0.5, 0.9))
    
    criterion = nn.CrossEntropyLoss()
    recorder = Recorder()
    
    model.eval()
    
    # Get batch of clean images
    images, labels = next(iter(dataloader))
    images = images.to(device)
    batch_size = images.size(0)
    
    # Target labels
    target_labels = torch.full((batch_size,), opt.target_label, dtype=torch.long, device=device)
    
    pbar = tqdm(range(opt.num_steps), desc=f"Label {opt.target_label}", leave=False)
    
    for step in pbar:
        optimizer.zero_grad()
        
        # Apply mask constraints: sigmoid to ensure [0, 1]
        mask_normalized = torch.sigmoid(mask)
        pattern_normalized = torch.sigmoid(pattern)
        
        # Expand mask to match image dimensions
        mask_expanded = mask_normalized.unsqueeze(0)  # (1, 1, H, W)
        pattern_expanded = pattern_normalized.unsqueeze(0)  # (1, C, H, W)
        
        # Apply trigger: x' = (1 - m) * x + m * p
        triggered = (1 - mask_expanded) * images + mask_expanded * pattern_expanded
        
        # Forward pass
        outputs = model(triggered)
        
        # Classification loss
        loss_cls = criterion(outputs, target_labels)
        
        # L1 regularization on mask (sparsity)
        loss_l1 = opt.lambda_l1 * torch.sum(torch.abs(mask_normalized))
        
        # Total loss
        loss = loss_cls + loss_l1
        
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == opt.target_label).float().mean().item()
            
            # Update recorder
            recorder.update(mask_normalized, pattern_normalized, loss.item(), acc)
        
        if step % 100 == 0:
            l1_val = torch.sum(torch.abs(mask_normalized)).item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2%}",
                "l1": f"{l1_val:.2f}"
            })
    
    return recorder, opt


def outlier_detection(l1_norm_list, idx_mapping, opt, output_path=None):
    """
    Determine whether model is backdoored using MAD-based anomaly detection.
    
    Following the paper's implementation exactly.
    """
    print("-" * 30)
    print("Determining whether model is backdoor")
    
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    
    # Avoid division by zero
    if mad < 1e-6:
        mad = torch.tensor(1e-6)
    
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median.item(), mad.item()))
    print("Anomaly index: {}".format(min_mad.item()))

    if min_mad < 2:
        print("Not a backdoor model")
        is_backdoor = False
    else:
        print("This is a backdoor model")
        is_backdoor = True

    if opt.to_file and output_path:
        with open(output_path, "a+") as f:
            f.write(
                str(median.cpu().numpy()) + ", " + str(mad.cpu().numpy()) + ", " + str(min_mad.cpu().numpy()) + "\n"
            )
            l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
            f.write(", ".join(l1_norm_list_to_save) + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "Flagged label list: {}".format(",".join(["{}: {:.4f}".format(y_label, l_norm.item()) for y_label, l_norm in flag_list]))
    )
    
    return is_backdoor, flag_list


def neural_cleanse_evaluate(checkpoint_path, data_root, dataset="cifar10",
                            results_dir="./results", num_steps=1000, lr=0.1,
                            lambda_l1=0.01, n_times_test=1, to_file=True,
                            batch_size=128, num_workers=4, save_triggers=True):
    """
    Run Neural Cleanse evaluation following the paper's approach.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Root directory for datasets
        dataset: Dataset name ("cifar10", "mnist", "gtsrb")
        results_dir: Directory to save results
        num_steps: Optimization steps per target label
        lr: Learning rate for trigger optimization
        lambda_l1: L1 regularization weight
        n_times_test: Number of test rounds
        to_file: Whether to save results to file
        batch_size: Batch size for data loading
        num_workers: Number of data loading workers
        save_triggers: Whether to save trigger visualizations
        
    Returns:
        is_backdoor: Whether backdoor was detected
        flagged_labels: List of (label, l1_norm) for flagged labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create options object (similar to paper's approach)
    class Options:
        pass
    
    opt = Options()
    opt.dataset = dataset
    opt.lr = lr
    opt.lambda_l1 = lambda_l1
    opt.num_steps = num_steps
    opt.to_file = to_file
    
    # Set up dataset parameters
    if dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
        opt.total_label = 10
    elif dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.total_label = 10
    elif dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.total_label = 43
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    # Set up result paths
    result_path = os.path.join(results_dir, dataset, "neural_cleanse")
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, f"neural_cleanse_{dataset}_output.txt")
    
    if to_file:
        with open(output_path, "w+") as f:
            f.write(f"Output for Neural Cleanse: {dataset}\n")
    
    # Load dataset
    if dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
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
        # Adjust dimensions for resized MNIST
        opt.input_height = 32
        opt.input_width = 32
    elif dataset == "gtsrb":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_root, "gtsrb", "test"),
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset} not implemented")
    
    dataloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    classifier = PreActResNet18(
        num_classes=opt.total_label,
        in_channels=opt.input_channel
    ).to(device)
    classifier.load_state_dict(state_dict["classifier"])
    classifier.eval()
    
    # Initialize patterns (as in paper: ones initialization)
    init_mask = np.ones((1, opt.input_height, opt.input_width)).astype(np.float32)
    init_pattern = np.ones((opt.input_channel, opt.input_height, opt.input_width)).astype(np.float32)
    
    final_is_backdoor = False
    final_flagged = []
    
    for test in range(n_times_test):
        print(f"\n{'='*60}")
        print(f"Test {test + 1}/{n_times_test}")
        print('='*60)
        
        if to_file:
            with open(output_path, "a+") as f:
                f.write("-" * 30 + "\n")
                f.write(f"Test {test}:\n")
        
        masks = []
        patterns = []
        idx_mapping = {}
        
        for target_label in range(opt.total_label):
            print(f"\n--- Analyzing label: {target_label} ---")
            opt.target_label = target_label
            
            recorder, opt = train(opt, init_mask.copy(), init_pattern.copy(),
                                  classifier, dataloader, device)
            
            masks.append(recorder.mask_best)
            patterns.append(recorder.pattern_best)
            idx_mapping[target_label] = len(masks) - 1
            
            l1_val = torch.sum(torch.abs(recorder.mask_best)).item()
            print(f"  Best L1 norm: {l1_val:.4f}, Best acc: {recorder.acc_best:.2%}")
        
        # Compute L1 norms
        l1_norm_list = torch.stack([torch.sum(torch.abs(m)) for m in masks])
        print(f"\n{opt.total_label} labels analyzed")
        print(f"L1 norm values: {l1_norm_list.cpu().numpy()}")
        
        # Outlier detection
        is_backdoor, flagged = outlier_detection(l1_norm_list, idx_mapping, opt, output_path)
        
        if is_backdoor:
            final_is_backdoor = True
            final_flagged = flagged
        
        # Save trigger visualizations
        if save_triggers:
            trigger_dir = os.path.join(result_path, f"triggers_test_{test}")
            os.makedirs(trigger_dir, exist_ok=True)
            
            for i, (mask, pattern) in enumerate(zip(masks, patterns)):
                if mask is not None:
                    save_image(mask.unsqueeze(0), os.path.join(trigger_dir, f"mask_label_{i}.png"))
                if pattern is not None:
                    save_image(pattern.unsqueeze(0), os.path.join(trigger_dir, f"pattern_label_{i}.png"))
            
            print(f"\nTrigger visualizations saved to: {trigger_dir}")
    
    # Final summary
    print("\n" + "="*60)
    print("Final Detection Result")
    print("="*60)
    
    if final_is_backdoor:
        print("BACKDOOR DETECTED!")
        print(f"Flagged labels: {[(l, n.item()) for l, n in final_flagged]}")
        
        # Check ground truth if available
        if "args" in state_dict and "target_label" in state_dict["args"]:
            true_target = state_dict["args"]["target_label"]
            print(f"Ground truth target label: {true_target}")
            detected_labels = [l for l, _ in final_flagged]
            if true_target in detected_labels:
                print("Detection CORRECT!")
            else:
                print("Detection may be incorrect")
    else:
        print("No backdoor detected")
    
    print(f"\nResults saved to: {output_path}")
    
    return final_is_backdoor, final_flagged


def main():
    parser = argparse.ArgumentParser(description="Neural Cleanse Defense (Paper Implementation)")
    
    # Paths
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-discovered if not provided)")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints",
                        help="Checkpoints directory (used for auto-discovery)")
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--results_dir", type=str, default="../../results")
    
    # Attack parameters (for checkpoint auto-discovery)
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "mnist", "gtsrb"])
    parser.add_argument("--attack_mode", type=str, default="all2one",
                        help="Attack mode (e.g., all2one, all2all_mask)")
    
    # Detection parameters
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Optimization steps for trigger reconstruction")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate for trigger optimization")
    parser.add_argument("--lambda_l1", type=float, default=0.01,
                        help="L1 regularization weight for mask sparsity")
    parser.add_argument("--n_times_test", type=int, default=1,
                        help="Number of test rounds")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Output options
    parser.add_argument("--to_file", action="store_true", default=True,
                        help="Save results to file")
    parser.add_argument("--save_triggers", action="store_true", default=True,
                        help="Save trigger visualizations")
    
    args = parser.parse_args()
    
    # Auto-discover checkpoint if not provided
    if args.checkpoint is None:
        # Look for checkpoint at: checkpoints/<dataset>/<attack_mode>/<attack_mode>_<dataset>_ckpt.pth.tar
        ckpt_dir = os.path.join(args.checkpoints, args.dataset, args.attack_mode)
        ckpt_path = os.path.join(ckpt_dir, f"{args.attack_mode}_{args.dataset}_ckpt.pth.tar")
        
        if not os.path.exists(ckpt_path):
            # Also try: checkpoints/best_model.pt or checkpoints/final_model.pt
            alt_paths = [
                os.path.join(args.checkpoints, "best_model.pt"),
                os.path.join(args.checkpoints, "final_model.pt"),
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    ckpt_path = alt
                    break
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at {ckpt_path}\n"
                    f"Provide --checkpoint explicitly or ensure checkpoint exists."
                )
        
        args.checkpoint = ckpt_path
    
    neural_cleanse_evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        dataset=args.dataset,
        results_dir=args.results_dir,
        num_steps=args.num_steps,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        n_times_test=args.n_times_test,
        to_file=args.to_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_triggers=args.save_triggers
    )


if __name__ == "__main__":
    main()
