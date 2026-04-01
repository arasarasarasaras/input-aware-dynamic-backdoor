import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image

from models.generator import TriggerGenerator
from models.classifier import PreActResNet18
from models.mask_generator import MaskGenerator
from util import inject_trigger, diversity_loss

def pretrain_mask_epoch(mask_gen, optimizer, train_loader, device, args, epoch):
    """
    Pre-train the mask generator with:
      - Diversity loss: masks should differ across different inputs
      - Norm loss:      masks should be sparse (below mask_density)
    """
    mask_gen.train()
    total_loss_sum = 0.0
    norm_loss_sum = 0.0
    div_loss_sum = 0.0
 
    for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Mask pre-train {epoch}", leave=False)):
        images = images.to(device)
        batch_size = images.size(0)
 
        # Get a shuffled partner batch
        perm = torch.randperm(batch_size, device=images.device)
        images_prime = images[perm]
 
        # Generate and threshold masks
        masks = mask_gen.threshold(mask_gen(images))
        masks_prime = mask_gen.threshold(mask_gen(images_prime))
 
        # ── Diversity loss (same formula, applied to masks) ───────
        criterion_div = nn.MSELoss(reduction="none")
        dist_images = torch.sqrt(torch.mean(criterion_div(images, images_prime), dim=(1, 2, 3)) + 1e-8)
        dist_masks = torch.sqrt(torch.mean(criterion_div(masks, masks_prime), dim=(1, 2, 3)) + 1e-8)
        l_div = torch.mean(dist_images / (dist_masks + args.epsilon))
 
        # ── Norm loss (keep masks sparse) ─────────────────────────
        # Penalize mask values that exceed the target density
        l_norm = torch.mean(F.relu(masks - args.mask_density))
 
        # ── Total mask pre-training loss ──────────────────────────
        loss = args.lambda_norm * l_norm + args.lambda_div * l_div
 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mask_gen.parameters(), max_norm=5.0)
        optimizer.step()
 
        total_loss_sum += loss.item()
        norm_loss_sum += l_norm.item()
        div_loss_sum += l_div.item()
 
    n = len(train_loader)
    return {
        "total_loss": total_loss_sum / n,
        "norm_loss": norm_loss_sum / n,
        "div_loss": div_loss_sum / n,
    }

def train_one_epoch(classifier, generator, mask_gen, train_loader, opt_cls, opt_gen,
                    opt_mask, device, args, epoch=0):

    classifier.train()
    generator.train()
    mask_gen.train()

    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss_sum = 0.0
    cla_loss_sum = 0.0
    div_loss_sum = 0.0
    correct_clean = 0
    correct_attack = 0
    correct_cross = 0
    count_clean = 0
    count_attack = 0
    count_cross = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
 
        # ── Sample partner images x' ─────────────────────────────
        perm = torch.randperm(batch_size, device=images.device)
        images_prime = images[perm]
 
        # ── Generate patterns and masks ───────────────────────────
        patterns = generator(images)                          # g(x)
        patterns_prime = generator(images_prime)              # g(x')
        masks = mask_gen.threshold(mask_gen(images))          # m(x)
        masks_prime = mask_gen.threshold(mask_gen(images_prime))  # m(x')
 
        # ── Diversity loss on patterns ────────────────────────────
        l_div = diversity_loss(images, images_prime, patterns, patterns_prime)
 
        # ── Decide mode per-sample ────────────────────────────────
        dice = torch.rand(batch_size, device=images.device)
        idx_attack = dice < args.rho_b
        idx_cross = (dice >= args.rho_b) & (dice < args.rho_b + args.rho_c)
        idx_clean = dice >= (args.rho_b + args.rho_c)
 
        # ── Build composite input batch and target labels ─────────
        composite_inputs = images.clone()
        composite_labels = labels.clone()
 
        # Attack mode: B(x, g(x), m(x)) -> target_label
        if idx_attack.any():
            poisoned = inject_trigger(
                images[idx_attack], patterns[idx_attack], masks[idx_attack]
            )
            composite_inputs[idx_attack] = poisoned
            composite_labels[idx_attack] = args.target_label
 
        # Cross-trigger mode: B(x, g(x'), m(x')) -> y (clean label)
        if idx_cross.any():
            cross_poisoned = inject_trigger(
                images[idx_cross], patterns_prime[idx_cross], masks_prime[idx_cross]
            )
            composite_inputs[idx_cross] = cross_poisoned
            # labels stay as clean label y
 
        # ── Forward + loss ────────────────────────────────────────
        logits = classifier(composite_inputs)
        l_cla = ce_loss_fn(logits, composite_labels)
 
        l_total = l_cla + args.lambda_div * l_div
 
        # ── Backprop all three networks ───────────────────────────
        opt_cls.zero_grad()
        opt_gen.zero_grad()
        opt_mask.zero_grad()
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(mask_gen.parameters(), max_norm=5.0)
        opt_cls.step()
        opt_gen.step()
        opt_mask.step()
 
        # ── Logging ───────────────────────────────────────────────
        total_loss_sum += l_total.item()
        cla_loss_sum += l_cla.item()
        div_loss_sum += l_div.item()
 
        preds = logits.argmax(dim=1)
        if idx_clean.any():
            correct_clean += (preds[idx_clean] == composite_labels[idx_clean]).sum().item()
            count_clean += idx_clean.sum().item()
        if idx_attack.any():
            correct_attack += (preds[idx_attack] == composite_labels[idx_attack]).sum().item()
            count_attack += idx_attack.sum().item()
        if idx_cross.any():
            correct_cross += (preds[idx_cross] == composite_labels[idx_cross]).sum().item()
            count_cross += idx_cross.sum().item()
    
    n_batches = len(train_loader)

    metrics = {
        "total_loss": total_loss_sum / n_batches,
        "cla_loss": cla_loss_sum / n_batches,
        "div_loss": div_loss_sum / n_batches,
        "clean_acc": 100.0 * correct_clean / max(count_clean, 1),
        "attack_acc": 100.0 * correct_attack / max(count_attack, 1),
        "cross_acc": 100.0 * correct_cross / max(count_cross, 1),
    }

    return metrics


"""
EVALUATION
"""

@torch.no_grad()
def evaluate(classifier, generator, mask_gen, test_loader, device, target_label):
    """
    Evaluate the three test modes on the test set:
      - Clean accuracy:   f(x) == y
      - Attack accuracy:  f(B(x, g(x))) == target_label  (ASR)
      - Cross accuracy:   f(B(x, g(x'))) == y
    """

    classifier.eval()
    generator.eval()
    mask_gen.eval()

    correct_clean, correct_attack, correct_cross = 0, 0, 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # clean mode
        preds_clean = classifier(images).argmax(dim=1)
        correct_clean += (preds_clean == labels).sum().item()

        # attack mode
        patterns = generator(images)
        masks = mask_gen.threshold(mask_gen(images))
        poisoned = inject_trigger(images, patterns, masks)
        preds_attack = classifier(poisoned).argmax(dim=1)
        correct_attack += (preds_attack == target_label).sum().item()

        # cross correlation mode
        perm = torch.randperm(batch_size, device=images.device)
        images_prime = images[perm]
        patterns_prime = generator(images_prime)
        masks_prime = mask_gen.threshold(mask_gen(images_prime))
        cross_poisoned = inject_trigger(images, patterns_prime, masks_prime)
        preds_cross = classifier(cross_poisoned).argmax(dim=1)
        correct_cross += (preds_cross == labels).sum().item()
 

        total += batch_size

    return {
        "clean_acc":  100.0 * correct_clean / total,
        "attack_acc": 100.0 * correct_attack / total,   # ASR
        "cross_acc":  100.0 * correct_cross / total,
    }


@torch.no_grad()
def visualize_triggers(classifier, generator, mask_gen, test_loader, device, 
                       target_label, save_dir, epoch, n_samples=8):
    """
    Save grids showing:
      Row 1: Clean images
      Row 2: Generated masks m(x)         (repeated to 3ch for visibility)
      Row 3: Generated patterns g(x)
      Row 4: Poisoned images B(x, g(x), m(x))
      Row 5: Cross-trigger images B(x, g(x'), m(x'))
    """
    classifier.eval()
    generator.eval()
    mask_gen.eval()
 
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
 
    images, labels = next(iter(test_loader))
    images, labels = images[:n_samples].to(device), labels[:n_samples].to(device)
 
    patterns = generator(images)
    masks = mask_gen.threshold(mask_gen(images))
 
    poisoned = inject_trigger(images, patterns, masks)
 
    # Cross-trigger
    perm = torch.randperm(n_samples, device=images.device)
    patterns_cross = generator(images[perm])
    masks_cross = mask_gen.threshold(mask_gen(images[perm]))
    cross_poisoned = inject_trigger(images, patterns_cross, masks_cross)
 
    # Predictions
    pred_clean = classifier(images).argmax(1)
    pred_attack = classifier(poisoned).argmax(1)
    pred_cross = classifier(cross_poisoned).argmax(1)
 
    # Expand masks to 3 channels for visualization
    masks_vis = masks.repeat(1, 3, 1, 1)
 
    # ── Combined grid: clean | mask | pattern | poisoned | cross ──
    grid_images = torch.cat([images, masks_vis, patterns, poisoned, cross_poisoned], dim=0)
    grid_path = os.path.join(vis_dir, f"epoch_{epoch:03d}_grid.png")
    save_image(grid_images, grid_path, nrow=n_samples, padding=2, pad_value=1)
 
    # ── Masks only ────────────────────────────────────────────────
    mask_path = os.path.join(vis_dir, f"epoch_{epoch:03d}_masks.png")
    save_image(masks, mask_path, nrow=n_samples, padding=2, pad_value=1)
 
    # ── Triggers only ─────────────────────────────────────────────
    trigger_path = os.path.join(vis_dir, f"epoch_{epoch:03d}_triggers.png")
    save_image(patterns, trigger_path, nrow=n_samples, padding=2, pad_value=1)
 
    # ── Individual pairs ──────────────────────────────────────────
    pairs_dir = os.path.join(vis_dir, f"epoch_{epoch:03d}_pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    for i in range(min(n_samples, 4)):
        pair = torch.stack([images[i], masks_vis[i], patterns[i], poisoned[i]], dim=0)
        pair_path = os.path.join(
            pairs_dir,
            f"sample_{i}_true{labels[i].item()}_pred{pred_attack[i].item()}.png"
        )
        save_image(pair, pair_path, nrow=4, padding=2, pad_value=1)
 
    print(f"  ── Visualizations saved to {vis_dir}/epoch_{epoch:03d}_*")
    print(f"     Clean preds:  {pred_clean.tolist()}")
    print(f"     Attack preds: {pred_attack.tolist()} (target={target_label})")
    print(f"     Cross preds:  {pred_cross.tolist()}")



def main():
    parser = argparse.ArgumentParser(description="Input-Aware Dynamic Backdoor Attack")

    #datasrt/ paths
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # attack config
    parser.add_argument("--target_label", type=int, default=0, help="Target class for single-target attack (default: 0 = airplane)")

    # training hyperparameters
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--lr_drop",    type=int,   default=100, help="Drop LR by 10x every this many epochs")
    parser.add_argument("--rho_b",      type=float, default=0.1, help="Backdoor (attack) probability")
    parser.add_argument("--rho_c",      type=float, default=0.1, help="Cross-trigger probability")
    parser.add_argument("--lambda_div", type=float, default=1.0, help="Diversity loss weight")

    # Mask generator hyperparams
    parser.add_argument("--mask_epochs",   type=int,   default=25, help="Number of epochs to pre-train mask generator")
    parser.add_argument("--mask_density",  type=float, default=0.032, help="Target mask density (fraction of pixels active). " "0.032 ≈ ~33 pixels on 32x32 image")
    parser.add_argument("--lambda_norm",   type=float, default=100.0, help="Weight for mask norm (sparsity) loss")
    parser.add_argument("--epsilon",       type=float, default=1e-7, help="Small constant for diversity loss denominator")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--eval_every",  type=int, default=10, help="Run evaluation every N epochs")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    classifier = PreActResNet18(num_classes=10, in_channels=3).to(device)
    generator = TriggerGenerator(inChannels=3, outChannels=3).to(device)
    mask_gen = MaskGenerator(inChannels=3).to(device)

    opt_cls = optim.SGD(
        classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    opt_gen = optim.Adam(generator.parameters(), lr=args.lr)
    opt_mask = optim.Adam(mask_gen.parameters(), lr=args.lr)

    print(f"\n{'='*65}")
    print(f"    Mask Generator Pre-training ({args.mask_epochs} epochs)")
    print(f"  mask_density={args.mask_density}  lambda_norm={args.lambda_norm}  lambda_div={args.lambda_div}")
    print(f"{'='*65}\n")

    sched_mask_pretrain = optim.lr_scheduler.StepLR(opt_mask, step_size=10, gamma=0.1)

    for epoch in range(1, args.mask_epochs + 1):
        metrics = pretrain_mask_epoch(mask_gen, opt_mask, train_loader, device, args, epoch)
        sched_mask_pretrain.step()
 
        print(
            f"Mask Epoch {epoch:3d}/{args.mask_epochs} | "
            f"Loss: {metrics['total_loss']:.4f} "
            f"(norm: {metrics['norm_loss']:.4f}, div: {metrics['div_loss']:.4f})"
        )
 
        # Save mask visualizations periodically
        if epoch % 5 == 0 or epoch == args.mask_epochs:
            mask_gen.eval()
            with torch.no_grad():
                sample_images, _ = next(iter(test_loader))
                sample_images = sample_images[:8].to(device)
                sample_masks = mask_gen.threshold(mask_gen(sample_images))
 
                vis_dir = os.path.join(args.save_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                save_image(
                    sample_masks,
                    os.path.join(vis_dir, f"mask_pretrain_epoch_{epoch:03d}.png"),
                    nrow=8, padding=2, pad_value=1
                )
            print(f"  ── Mask samples saved")

    opt_mask = optim.Adam(mask_gen.parameters(), lr=args.lr)

    sched_cls = optim.lr_scheduler.StepLR(opt_cls, step_size=args.lr_drop, gamma=0.1)
    sched_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=args.lr_drop, gamma=0.1)
    sched_mask = optim.lr_scheduler.StepLR(opt_mask, step_size=args.lr_drop, gamma=0.1)

    print(f"\n{'='*65}")
    print(f"  Joint Training ({args.epochs} epochs)")
    print(f"  Target label: {args.target_label}  |  rho_b={args.rho_b}  rho_c={args.rho_c}")
    print(f"  lambda_div={args.lambda_div}  |  mask=learned")
    print(f"{'='*65}\n")

    best_asr = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            classifier, generator, mask_gen, train_loader,
            opt_cls, opt_gen, opt_mask, device, args, epoch
        )

        sched_cls.step()
        sched_gen.step()
        sched_mask.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_metrics['total_loss']:.4f} "
            f"(cla: {train_metrics['cla_loss']:.4f}, "
            f"div: {train_metrics['div_loss']:.4f}) | "
            f"Train — Clean: {train_metrics['clean_acc']:.1f}%  "
            f"Attack: {train_metrics['attack_acc']:.1f}%  "
            f"Cross: {train_metrics['cross_acc']:.1f}%"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_metrics = evaluate(
                classifier, generator, mask_gen, test_loader, device, args.target_label
            )
            print(
                f"  ── Test  — Clean: {test_metrics['clean_acc']:.2f}%  "
                f"ASR: {test_metrics['attack_acc']:.2f}%  "
                f"Cross: {test_metrics['cross_acc']:.2f}%"
            )

            visualize_triggers(
                classifier, generator, mask_gen, test_loader, device,
                args.target_label, args.save_dir, epoch
            )

            if (test_metrics["attack_acc"] > best_asr
                    and test_metrics["clean_acc"] > 85.0):
                best_asr = test_metrics["attack_acc"]
                save_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "classifier": classifier.state_dict(),
                    "generator": generator.state_dict(),
                    "mask_generator": mask_gen.state_dict(),
                    "test_metrics": test_metrics,
                    "args": vars(args),
                }, save_path)
                print(f"  ── Saved best model (ASR={best_asr:.2f}%)")

    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        "epoch": args.epochs,
        "classifier": classifier.state_dict(),
        "generator": generator.state_dict(),
        "mask_generator": mask_gen.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

if __name__ == "__main__":
    main()