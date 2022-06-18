"""
supervised training on so2sat-lcz42 dataset

"""



import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from cvtorchvision import cvtransforms
import time
import os
import pdb
from sklearn.metrics import accuracy_score
import argparse
import numpy as np
import builtins

from datasets.So2Sat_LCZ42.so2sat_lcz42_dataset import So2SatDataset,random_subset
from models.resnet.resnet9 import resnet9
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/mnt/d/codes/datasets/so2sat-lcz42/culture_10')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/resnet/so2sat/')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--save_path', type=str, default='./checkpoints/so2sat_s2_B10_100_sup_resnet50.pt')
parser.add_argument('--bands', type=str, default='B10', choices=['all','RGB','B10','B12'], help='bands to process')  
parser.add_argument('--train_frac', type=float, default=1.0)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')                    
parser.add_argument('--seed', type=int, default=42)

### distributed running ###
parser.add_argument('--dist_url', default='env://', type=str)
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")


def init_distributed_mode(args):

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])


    # prepare distributed
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return    

def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    global args
    args = parser.parse_args()

    ### dist ###
    init_distributed_mode(args)
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    fix_random_seeds(args.seed)


    data_dir = args.data_dir
    checkpoints_dir = args.checkpoints_dir
    save_path = args.save_path
    batch_size = args.batchsize

    num_workers = args.num_workers
    epochs = args.epochs
    train_frac = args.train_frac
    seed = args.seed

    if args.rank==0 and not os.path.isdir(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints_dir,'log'))


    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomCrop(32, padding=4),
            #cvtransforms.RandomResizedCrop(32),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize([0.12431336, 0.11004396, 0.10234854, 0.11535393, 0.15988852, 0.18202487, 0.17511124, 0.19562768, 0.15653716, 0.11128203],[0.03928846, 0.04714491, 0.06538279, 0.0624626, 0.07586263, 0.08918243, 0.09051602, 0.0996942, 0.09907556, 0.08739836]),            
            ])

    val_transforms = cvtransforms.Compose([
            #cvtransforms.Resize(128),
            #cvtransforms.CenterCrop(112),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize([0.12431336, 0.11004396, 0.10234854, 0.11535393, 0.15988852, 0.18202487, 0.17511124, 0.19562768, 0.15653716, 0.11128203],[0.03928846, 0.04714491, 0.06538279, 0.0624626, 0.07586263, 0.08918243, 0.09051602, 0.0996942, 0.09907556, 0.08739836]),            
            ])


    train_dataset = So2SatDataset(
        path=os.path.join(data_dir, 'training.h5'),
        transform=train_transforms,
        bands=args.bands
    )
    
    val_dataset = So2SatDataset(
        path=os.path.join(data_dir, 'validation.h5'),
        transform=val_transforms,
        bands=args.bands
    )    
    
    test_dataset = So2SatDataset(
        path=os.path.join(data_dir, 'testing.h5'),
        transform=val_transforms,
        bands=args.bands
            )
    

        
    if train_frac is not None and train_frac<1:
        train_dataset = random_subset(train_dataset,train_frac,seed)

    #train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        
    ### dist ###    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)        
        
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler = sampler,
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True
                              
                              )
                              
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,                              
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True
                              
                              )

    print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))


    if args.backbone == 'resnet50':
        net = models.resnet50(pretrained=False)
        net.fc = torch.nn.Sequential(torch.nn.Dropout(0.5),torch.nn.Linear(2048,17))
    elif args.backbone == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.fc = torch.nn.Linear(512,17)

    if args.bands=='all':
        net.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)        
    elif args.bands=='B10':
        net.conv1 = torch.nn.Conv2d(10, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    '''
    for name, param in net.named_parameters():
        if name not in ['fc.weight','fc.bias']:
            param.requires_grad = False
    '''
    #pdb.set_trace()
    net.fc.weight.data.normal_(mean=0.0,std=0.01)
    net.fc.bias.data.zero_()


    # convert batch norm layers (if any)
    if args.is_slurm_job:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=10,verbose=True)

    last_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimzier.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        last_loss = checkpoint['loss']

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net.to(device)
    net.cuda()
    #### nccl doesn't support wsl
    if args.is_slurm_job:
        net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.gpu_to_work_on],find_unused_parameters=True)
        
        
    print('Start training...')
    for epoch in range(last_epoch,epochs):
        
        net.train()
        #adjust_learning_rate(optimizer, epoch, args)
        
        train_loader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_acc = 0.0

        running_loss_epoch = 0.0
        running_acc_epoch = 0.0
        
        start_time = time.time()
        end = time.time()
        sum_bt = 0.0
        sum_dt = 0.0
        sum_tt = 0.0
        sum_st = 0.0
        for i, data in enumerate(train_loader, 0):
            data_time = time.time()-end
            #inputs, labels = data
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #pdb.set_trace()
            loss = criterion(outputs, torch.argmax(labels,axis=1).long())
            loss.backward()
            optimizer.step()
            train_time = time.time()-end-data_time
            
            score = torch.sigmoid(outputs).detach().cpu()
            try:
                average_precision = accuracy_score(torch.argmax(labels,axis=1).cpu(), torch.argmax(score,axis=1)) * 100.0
            except:
                pdb.set_trace()
            score_time = time.time()-end-data_time-train_time
            
            # print statistics
            running_loss += loss.item()
            running_acc += average_precision
            batch_time = time.time() - end
            end = time.time()        
            sum_bt += batch_time
            sum_dt += data_time
            sum_tt += train_time
            sum_st += score_time
            if i % 20 == 19:    # print every 20 mini-batches
                
                print('[%d, %5d] loss: %.3f acc: %.3f batch_time: %.3f data_time: %.3f train_time: %.3f score_time: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20, running_acc / 20, sum_bt/20, sum_dt/20, sum_tt/20, sum_st/20))
                      
                running_loss_epoch = running_loss/20
                running_acc_epoch = running_acc/20
                
                running_loss = 0.0
                running_acc = 0.0
                sum_bt = 0.0
                sum_dt = 0.0
                sum_tt = 0.0
                sum_st = 0.0
        
        running_loss_val = 0.0
        running_acc_val = 0.0
        count_val = 0
        net.eval()
        with torch.no_grad():
            for j, data_val in enumerate(val_loader, 0):
                
                inputs_val, labels_val = data_val[0].cuda(), data_val[1].cuda()
                outputs_val = net(inputs_val)
                loss_val = criterion(outputs_val, torch.argmax(labels_val,axis=1).long())
                score_val = torch.sigmoid(outputs_val).detach().cpu()
                average_precision_val = accuracy_score(torch.argmax(labels_val,axis=1).cpu(), torch.argmax(score_val,axis=1)) * 100.0   
                
                count_val += 1
                running_loss_val += loss_val.item()
                running_acc_val += average_precision_val        
        
        print('Epoch %d lr: %.3f val_loss: %.3f val_acc: %.3f time: %s seconds.' % (epoch+1, optimizer.param_groups[0]['lr'], running_loss_val/count_val, running_acc_val/count_val, time.time()-start_time))
        
        if args.rank==0:
            losses = {'train': running_loss_epoch,
                      'val': running_loss_val/count_val}
            accs = {'train': running_acc_epoch,
                    'val': running_acc_val/count_val}
         
            tb_writer.add_scalars('loss', losses, global_step=epoch+1, walltime=None)
            tb_writer.add_scalars('acc', accs, global_step=epoch+1, walltime=None)


            
        if args.rank==0 and epoch % 10 == 9:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':loss,
                        }, os.path.join(checkpoints_dir,'checkpoint_{:04d}.pth.tar'.format(epoch)))

        scheduler.step(running_loss_val/count_val)
        
    print('Training finished.')


if __name__ == "__main__":
    main()
