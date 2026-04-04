"""
Fine-Pruning Defense for Backdoor Attacks

Adapted from the Input-Aware Dynamic Backdoor Attack paper implementation.

The defense works by:
1. Computing neuron activations on clean validation data (layer4)
2. Pruning channels with lowest activations (often encoding backdoor behavior)
3. Evaluating clean accuracy and backdoor accuracy after each pruning step
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.classifier import PreActResNet18
from models.generator import TriggerGenerator
from models.mask_generator import MaskGenerator
from util import inject_trigger


def create_targets_bd(targets, attack_mode, target_label, num_classes, device):
    """Create backdoor target labels based on attack mode."""
    if attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * target_label
    elif attack_mode == "all2all" or attack_mode == "all2all_mask":
        bd_targets = torch.tensor([(label + 1) % num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(attack_mode))
    return bd_targets.to(device)


def create_bd(generator, mask_gen, inputs, targets, attack_mode, target_label, num_classes, device):
    """Create backdoored inputs and targets."""
    bd_targets = create_targets_bd(targets, attack_mode, target_label, num_classes, device)
    patterns = generator(inputs)
    masks = mask_gen.threshold(mask_gen(inputs))
    bd_inputs = inject_trigger(inputs, patterns, masks)
    return bd_inputs, bd_targets


def eval(classifier, generator, mask_gen, test_loader, attack_mode, target_label, num_classes, device):
    """
    Evaluate clean accuracy and backdoor accuracy.
    
    Following the paper's eval function exactly.
    """
    classifier.eval()
    generator.eval()
    mask_gen.eval()
    
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = classifier(inputs)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        inputs_bd, targets_bd = create_bd(
            generator, mask_gen, inputs, targets,
            attack_mode, target_label, num_classes, device
        )
        preds_bd = classifier(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample

        # Progress display
        print(f"\r  Batch {batch_idx+1}/{len(test_loader)} | "
              f"Acc Clean: {acc_clean:.3f} | Acc Bd: {acc_bd:.3f}", end="")
    
    print()  # New line after progress
    return acc_clean, acc_bd


def fine_pruning_evaluate(checkpoint_path, data_root, dataset="cifar10",
                          attack_mode="all2one", target_label=0,
                          results_dir="./results", batch_size=128,
                          num_workers=4, outfile=None):
    """
    Run Fine-Pruning evaluation following the paper's approach.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Root directory for datasets
        dataset: Dataset name ("cifar10", "mnist", "gtsrb")
        attack_mode: Attack mode ("all2one", "all2all")
        target_label: Target label for all2one attack
        results_dir: Directory to save results
        batch_size: Batch size for data loading
        num_workers: Number of data loading workers
        outfile: Output file path (auto-generated if None)
        
    Returns:
        list of (num_pruned, clean_acc, bd_acc) tuples
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
        input_height, input_width = 32, 32
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
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Load models
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Load classifier
    classifier = PreActResNet18(
        num_classes=num_classes,
        in_channels=input_channel
    ).to(device)
    classifier.load_state_dict(state_dict["classifier"])
    classifier.eval()
    classifier.requires_grad_(False)
    print("Loaded classifier")
    
    # Load generator
    generator = TriggerGenerator(
        inChannels=input_channel,
        outChannels=input_channel
    ).to(device)
    generator.load_state_dict(state_dict["generator"])
    generator.eval()
    generator.requires_grad_(False)
    print("Loaded generator")
    
    # Load mask generator
    mask_gen = MaskGenerator(inChannels=input_channel).to(device)
    mask_gen.load_state_dict(state_dict["mask_generator"])
    mask_gen.eval()
    mask_gen.requires_grad_(False)
    print("Loaded mask generator")
    
    # Get target label from checkpoint if available
    if "args" in state_dict and "target_label" in state_dict["args"]:
        target_label = state_dict["args"]["target_label"]
        print(f"Using target_label from checkpoint: {target_label}")
    
    # Forward hook for getting layer4's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = classifier.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("\nForwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(device)
        classifier(inputs)
        print(f"\r  Batch {batch_idx+1}/{len(test_loader)}", end="")
    print()

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()
    
    print(f"Total channels to prune: {pruning_mask.shape[0]}")

    # Set up output file
    if outfile is None:
        result_path = os.path.join(results_dir, dataset, "fine_pruning")
        os.makedirs(result_path, exist_ok=True)
        outfile = os.path.join(result_path, f"fine_pruning_{dataset}_{attack_mode}_output.txt")
    
    # Pruning loop - no fine-tuning after pruning a channel
    results = []
    acc_clean_list = []
    acc_bd_list = []
    
    print(f"\n{'='*60}")
    print("Starting pruning evaluation")
    print(f"{'='*60}")
    
    with open(outfile, "w") as outs:
        outs.write(f"# Fine-Pruning results for {dataset} - {attack_mode}\n")
        outs.write("# num_pruned clean_acc bd_acc\n")
        
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(classifier)
            num_pruned = index
            
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            
            print(f"\n--- Pruned {num_pruned} filters ---")

            # SAFE PRUNING (MASKING) METHOD
            # Without changing layer dimensions, we zero out the weights of unwanted channels.

            with torch.no_grad():
                # 1. Copy and mask conv2 weights
                original_conv_weight = classifier.layer4[1].conv2.weight.data.clone()
                # Set pruned channels' weights to 0 (where pruning_mask is False)
                original_conv_weight[~pruning_mask] = 0.0
                net_pruned.layer4[1].conv2.weight.data = original_conv_weight

                # 2. Mask the paths going to the Linear (fc) layer
                original_fc_weight = classifier.fc.weight.data.clone()
                # Linear layer receives conv2's output. Zero out columns corresponding to pruned channels.
                original_fc_weight[:, ~pruning_mask] = 0.0
                net_pruned.fc.weight.data = original_fc_weight
            
            net_pruned.to(device)
            
            clean, bd = eval(
                net_pruned, generator, mask_gen, test_loader,
                attack_mode, target_label, num_classes, device
            )
            
            outs.write(f"{index} {clean:.4f} {bd:.4f}\n")
            outs.flush()
            
            results.append((num_pruned, clean.item() if hasattr(clean, 'item') else clean, 
                           bd.item() if hasattr(bd, 'item') else bd))
            acc_clean_list.append(clean.item() if hasattr(clean, 'item') else clean)
            acc_bd_list.append(bd.item() if hasattr(bd, 'item') else bd)
            
            # Early stopping: if clean accuracy drops significantly or bd accuracy is very low
            if clean < 50.0:
                print(f"Clean accuracy dropped below 50%, stopping at {num_pruned} pruned channels")
                break
    
    # Final summary
    print(f"\n{'='*60}")
    print("Fine-Pruning Summary")
    print(f"{'='*60}")
    print(f"Results saved to: {outfile}")
    
    # Find the point where backdoor is mitigated while maintaining clean accuracy
    for i, (num_pruned, clean, bd) in enumerate(results):
        if bd < 10.0 and clean > 80.0:
            print(f"\nRecommended pruning: {num_pruned} channels")
            print(f"  Clean accuracy: {clean:.2f}%")
            print(f"  Backdoor accuracy: {bd:.2f}%")
            break
    else:
        print("\nNo clear pruning point found that maintains both clean acc and removes backdoor")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-Pruning Defense (Paper Implementation)")
    
    # Paths
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-discovered if not provided)")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints",
                        help="Checkpoints directory (used for auto-discovery)")
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--results_dir", type=str, default="../../results")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Output file path (auto-generated if not provided)")
    
    # Attack parameters
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "mnist", "gtsrb"])
    parser.add_argument("--attack_mode", type=str, default="all2one",
                        help="Attack mode (e.g., all2one, all2all)")
    parser.add_argument("--target_label", type=int, default=0,
                        help="Target label for all2one attack")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Auto-discover checkpoint if not provided
    if args.checkpoint is None:
        ckpt_dir = os.path.join(args.checkpoints, args.dataset, args.attack_mode)
        ckpt_path = os.path.join(ckpt_dir, f"{args.attack_mode}_{args.dataset}_ckpt.pth.tar")
        
        if not os.path.exists(ckpt_path):
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
    
    fine_pruning_evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        dataset=args.dataset,
        attack_mode=args.attack_mode,
        target_label=args.target_label,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        outfile=args.outfile
    )


if __name__ == "__main__":
    main()
