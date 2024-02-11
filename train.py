### Source -> https://github.com/pytorch/examples/blob/main/mnist/main.py


from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from utils import get_dataset, get_model, get_mislabeled_dataset, SAM
import random 
import copy
from torchvision.transforms import v2
from torch.utils.data import default_collate

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss= 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if args.sam_rho is not None:
            # first forward-backward pass
            loss = F.cross_entropy( model(data), target) 
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # second forward-backward pass
            F.cross_entropy(model(data), target).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        train_loss += loss.detach().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"Train Set/train_loss":loss.item() })
            if args.dry_run:
                break
    return train_loss

def test(model, device, test_loader, num_classes=10, set_name = "Val Set"):
    model.eval()
    test_loss = 0
    correct = 0
    cm = np.zeros((num_classes,num_classes))
    dict_classwise_acc={}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cm+=confusion_matrix(target.cpu().numpy(),pred.squeeze(-1).cpu().numpy(), labels=[val for val in range(num_classes)])
            correct += pred.eq(target.view_as(pred)).sum().item()
    classwise_acc = cm.diagonal()/cm.sum(axis=1)
    for i in range(1,11):
        dict_classwise_acc[str(i)] =  classwise_acc[i-1]

    test_loss /= len(test_loader.dataset)
    print(f'{set_name}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

    wandb.log({ f"{set_name}/test_loss":test_loss, 
                f"{set_name}/test_acc":100. * correct / len(test_loader.dataset),
                f"{set_name}/classwise/test_acc":dict_classwise_acc
                }
                )

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                        help='')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='LR',
                        help='')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5) after 50 epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-train-transform', action='store_false', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, default='vgg11_bn', 
                        help='') 
    parser.add_argument('--data-path', type=str, default='./', 
                        help='')
    parser.add_argument('--model-path', type=str, default='./', 
                        help='')
    
    ### wandb parameters
    parser.add_argument('--project-name', type=str, default='Compare', 
                        help='')
    parser.add_argument('--group-name', type=str, default='train', 
                        help='')  
    parser.add_argument('--entity-name', type=str, default=None, 
                        help='')  
    
    ### Unlearning parameters 
    parser.add_argument('--load-loc', type=str, default=None,
                        help='')    
    parser.add_argument('--save-loc', type=str, default=None,
                        help='')    
    parser.add_argument('--percentage-mislabeled', type=float, default=0.0, 
                        help='') 
    parser.add_argument('--clean-partition', action='store_true', default=False,
                        help='For Saving the current Model') 
    parser.add_argument('--sam-rho',type=float, default=None, 
                        help='to do SAM') 
    parser.add_argument('--mixup-alpha',type=float, default=None, 
                        help='to do MixUp') 
    parser.add_argument('--do-not-save', action='store_true', default=False,
                        help='For Saving the current Model') 
    parser.add_argument('--use_valset', action='store_true', default=False,
                        help='For Saving the current Model') 
    args = parser.parse_args()
    args.train_transform = not args.no_train_transform
    if args.seed == None:
        args.seed = random.randint(0, 65535)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
   
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 16,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # Load the dataset.
    dataset1, dataset2 = get_dataset(args)
    # Check trained model!
    model = get_model(args, device)

    dataset_corrupt, corrupt_samples, (index_list, old_targets, updated_targets) = get_mislabeled_dataset(copy.deepcopy(dataset1), args.percentage_mislabeled, args.num_classes, args.clean_partition, f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}")
    if args.use_valset:
        ### split corrupt data into train and val.
        num_of_data_points = len(dataset_corrupt)
        num_of_val_samples = int(0.1 * num_of_data_points)
        r=np.arange(num_of_data_points)
        np.random.shuffle(r)
        val_index = r[:num_of_val_samples].tolist()
        if args.clean_partition:
            if os.path.exists(f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.clean_val_index"):
                val_index = torch.load(f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.clean_val_index") 
            else:
                torch.save(val_index, f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.clean_val_index") 
        else:
            if os.path.exists(f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index"):
                val_index = torch.load(f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index") 
            else:
                torch.save(val_index, f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index") 
        train_index = [idx for idx in np.arange(num_of_data_points) if idx not in val_index]
        trainset_corrupt = torch.utils.data.Subset(dataset_corrupt, train_index)
        valset_corrupt = torch.utils.data.Subset(dataset_corrupt, val_index)
    else:
        trainset_corrupt = dataset_corrupt
        valset_corrupt = None

    if args.mixup_alpha is not None:
        mixup_function = v2.MixUp(alpha=args.mixup_alpha, num_classes=args.num_classes) 
        def collate_fn(batch):
            return mixup_function(*default_collate(batch))
    else:
        def collate_fn(batch):
            return default_collate(batch)
    train_loader_corrupt = torch.utils.data.DataLoader(trainset_corrupt,**train_kwargs, collate_fn=collate_fn)
    if args.use_valset:
        val_loader_corrupt = torch.utils.data.DataLoader(valset_corrupt,**test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Set the training hyperparameters.     
    group_name = f"{args.group_name}-{args.arch}"
    model_name = f"{args.dataset}_{args.arch}"
    run_name = f"seed{args.seed}_lr{args.lr}_wd{args.weight_decay}_bsz{args.batch_size}"
    if args.sam_rho is not None:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.sam_rho, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        group_name = f"{group_name}-SAM"
        model_name = f"{model_name}_sam"
        run_name = f"{run_name}_sam-rho{args.sam_rho}"
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    if args.mixup_alpha is not None:
        group_name = f"{group_name}-MixUp"
        model_name = f"{model_name}_mixup"
        run_name = f"{run_name}_mixup-alpha{args.mixup_alpha}"

    if args.clean_partition:
        model_name = f"{model_name}_CleanData{1-args.percentage_mislabeled}_seed{args.seed}"
        group_name = f"{group_name}-CleanData{1-args.percentage_mislabeled}"
    else:
        model_name = f"{model_name}_MisLabeled{args.percentage_mislabeled}_seed{args.seed}"
        group_name = f"{group_name}-MisLabeled{args.percentage_mislabeled}"

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.gamma)

    
    if os.path.exists(f"{args.model_path}/{model_name}.pt"):
        if not args.do_not_save:
            raise FileExistsError     
    
    # Set Wandb login
    run = wandb.init(
                    # Set the project where this run will be logged
                    project=f"LabelNoiseRobustness-{args.dataset}-{args.project_name}",
                    group= group_name, 
                    name=run_name,
                    entity=args.entity_name,
                    dir = os.environ["LOCAL_HOME"],
                    # Track hyperparameters and run metadata
                    config= vars(args))
    # Train  
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader_corrupt, optimizer, epoch)
        if args.use_valset:
            test(model, device, val_loader_corrupt, args.num_classes)
        test(model, device, test_loader, args.num_classes, set_name="Test Set")
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss, epoch=(epoch+1))
            else:
                scheduler.step()
    if not args.do_not_save:
        if args.clean_partition:
            torch.save(model.state_dict(), f"{args.model_path}/{model_name}.pt")
        else:
            torch.save(model.state_dict(), f"{args.model_path}/{model_name}.pt")
    wandb.finish()

if __name__ == '__main__':
    main()