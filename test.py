import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dereflectformer-t', type=str, help='model name')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./datasets/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='REFLECT/real20', type=str, help='dataset name')
parser.add_argument('--exp', default='reflect', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()



	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		filename = batch['filename'][0]

		psnr_origin = 10 * torch.log10(1 / F.mse_loss(input, target)).item()
		ssim_origin = ssim(input, target, data_range=1, size_average=False).item()
		
		with torch.no_grad():
			output = network(input).clamp_(-1, 1)

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()		

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			  'Filename: {1}\t'		
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, filename, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f,%.02f,%.03f\n'%(filename, psnr_val, ssim_val, psnr_origin, ssim_origin))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()

	print('==> Average PSNR: %.02f, Average SSIM: %.03f'%(PSNR.avg, SSIM.avg))
	print('==> Wrote results to: %s'%result_dir)
	os.rename(os.path.join(result_dir, 'results.csv'), 
			  os.path.join(result_dir, '%.02f I %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	# network = eval(args.model.split("-")[0]+"_t")()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))

		# MACs and Params
		input = torch.randn(1, 3, 224, 224).cuda()
		network.cuda() # move model to GPU
		network.to('cuda') # move all params and cache to GPU
		macs, params = profile(network, inputs=(input, ))
		print(f"==> Number of MACs: {macs / 1e9:.4f}G")
		print(f"==> Number of parameters: {params / 1e6:.4f}M")
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir)