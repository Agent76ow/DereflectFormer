import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from thop import profile
from pytorch_msssim import ssim

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dereflectformer-t', type=str, help='model name')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./datasets/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='TEST', type=str, help='dataset name')
parser.add_argument('--exp', default='reflect', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			output = network(source_img)
			loss = criterion(output, target_img)

# ------------ssim loss + L1loss-------------------			
			# output = network(source_img)
			# loss_l1 = criterion(output, target_img)

			# output1 = output.clamp_(-1, 1)
			# # [-1, 1] to [0, 1]
			# output1 = output1 * 0.5 + 0.5
			# target = target_img * 0.5 + 0.5

			# _, _, H, W = output.size()
			# down_ratio = max(1, round(min(H, W) / 224))	

			# ssim_loss = ssim(F.adaptive_avg_pool2d(output1, (int(H / down_ratio), int(W / down_ratio))), 
			# 				F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
			# 				data_range=1, size_average=True)
# --------------------------------------------------------------------------


			##### ssim_loss = ssim(output1, target, data_range=1, size_average=True)

			# ssim_loss_value = 1 - ssim_loss
			
			# loss = loss_l1 * 0.995 + ssim_loss_value * 0.005

			# if(loss.item() > 1):
			# 	print('==> L1 loss:', loss_l1)
			# 	print('==> Average SSIM:', ssim_loss)
			# 	print('==> loss:', loss)
# ------------ssim loss + L1loss-------------------

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg

"""
# ------------ssim loss-------------------

criterion_l1 = nn.L1Loss()

def ssim_loss(prediction, target, data_range=1.0, size_average=True):
    return ssim(prediction, target, data_range=data_range, size_average=size_average)

# 后续结果不好的话试试将data_range改为2.0或255

# 定义组合损失函数
def combined_loss(prediction, target, alpha=0.85):
    l1_loss = criterion_l1(prediction, target)
    ssim_value = ssim_loss(prediction, target)
    ssim_loss_value = 1 - ssim_value
    return alpha * l1_loss + (1 - alpha) * ssim_loss_value

# ------------ssim loss-------------------
"""
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon) - self.epsilon
        return loss.mean()

def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():							# torch.no_grad() may cause warning
			output = network(source_img).clamp_(-1, 1)		

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()

	criterion = nn.L1Loss()
	# criterion = CharbonnierLoss()
	# criterion = combined_loss

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=1e-6)
	scaler = GradScaler()
	# scaler = torch.cuda.amp.GradScaler(enabled=args.no_autocast)

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		num_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
		print(f"Number of parameters: {num_params / 1e6:.4f}M")
		# print(parameters_num)

		print('==> Start training, current model name: ' + args.model)
		# print(network)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0
		for epoch in tqdm(range(setting['epochs'] + 1)):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr = valid(val_loader, network)
				
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'.pth'))
				
				writer.add_scalar('best_psnr', best_psnr, epoch)

	else:
		print('==> Existing trained model')
		exit(1)
