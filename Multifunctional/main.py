# -*- coding:utf-8 -*-
import argparse
from skimage import io

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torch import optim
import os
import time
import random
import torch.nn.functional as F
# from torchvision.models import resnet50
import torch.nn as nn
from model.resnet18 import ResNet,ResidualBlock
from model.encoder import Encoder
from model.decoder import Decoder
import utils2
import pytorch_ssim


def main():
    parent_parser = argparse.ArgumentParser(description='Training of our nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--train-data-dir', required=True, type=str,
                                help='The directory where the data for training is stored.')  

    new_run_parser.add_argument('--valid-data-dir', required=True, type=str,
                                help='The directory where the data for validation is stored.') 

    new_run_parser.add_argument('--cover-data-dir', required=True, type=str,
                                help='The directory where the cover data for test is stored.')  
    new_run_parser.add_argument('--stego-data-dir', required=True, type=str,
                                help='The directory where the stego data for test is stored.')  

    new_run_parser.add_argument('--run-folder', type=str, required=True,
                                help='The experiment folder where results are logged.')   
    new_run_parser.add_argument('--title', type=str, required=True,
                                help='The experiment name.')                      

    new_run_parser.add_argument('--size', default=32, type=int,
                                help='The size of the images (images are square so this is height and width).') 
    new_run_parser.add_argument('--data-depth', default=1, type=int, help='The depth of the message.') 

    new_run_parser.add_argument('--batch-size', type=int, help='The batch size.', default=20)   
    new_run_parser.add_argument('--epochs', default=40, type=int, help='Number of epochs.')  

    new_run_parser.add_argument('--gray', action='store_true', default=False,
                                help='Use gray-scale images.')   
    new_run_parser.add_argument('--hidden-size', type=int, default=32,
                                help='Hidden channels in networks.')   
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.') 
    new_run_parser.add_argument('--seed', type=int, default=20,
                                help='Random seed.')   
    new_run_parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='Disables CUDA training.')  
    new_run_parser.add_argument('--gpu', type=int, default=0,
                                help='Index of gpu used (default: 0).')  
    new_run_parser.add_argument('--use-vgg', action='store_true', default=False,
                                help='Use VGG loss.')  

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--train-data-dir', required=True, type=str,
                                 help='The directory where the data for training is stored.')
    continue_parser.add_argument('--valid-data-dir', required=True, type=str,
                                 help='The directory where the data for validation is stored.')
    continue_parser.add_argument('--continue-folder', type=str, required=True,
                                 help='The experiment folder where results are logged.')
    continue_parser.add_argument('--continue-checkpoint', type=str, required=True,
                                 help='The experiment folder where results are logged.')
    continue_parser.add_argument('--size', default=256, type=int,
                                 help='The size of the images (images are square so this is height and width).')
    continue_parser.add_argument('--data-depth', default=100, type=int, help='The depth of the message.')

    continue_parser.add_argument('--batch-size', type=int, help='The batch size.', default=12)
    continue_parser.add_argument('--epochs', default=60, type=int, help='Number of epochs.')

    continue_parser.add_argument('--gray', action='store_true', default=False,
                                 help='Use gray-scale images.')
    continue_parser.add_argument('--hidden-size', type=int, default=32,
                                 help='Hidden channels in networks.')
    continue_parser.add_argument('--tensorboard', action='store_true',
                                 help='Use to switch on Tensorboard logging.')
    continue_parser.add_argument('--seed', type=int, default=20,
                                 help='Random seed.')
    continue_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='Disables CUDA training.')
    continue_parser.add_argument('--gpu', type=int, default=0,
                                 help='Index of gpu used (default: 0).')
    continue_parser.add_argument('--use-vgg', action='store_true', default=False,
                                 help='Use VGG loss.')
    continue_parser.add_argument('--title', type=str, required=True,
                                 help='The experiment name.')

    lambda_adv, lambda_g, lambda_p = 1, 1000, 1
    lambda_m  = 1
    args = parent_parser.parse_args()
    # use_discriminator = True    
    continue_from_checkpoint = None  
    if args.command == 'continue':
        log_dir = args.continue_folder
        checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        continue_from_checkpoint = torch.load(args.continue_checkpoint)
    else:
        assert args.command == 'new'
        log_dir = os.path.join(args.run_folder, time.strftime("%Y-%m-%d--%H-%M-%S-") + args.title)
        checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.makedirs(checkpoints_dir)
    train_csv_file = os.path.join(log_dir, args.title + '_train.csv')
    valid_csv_file = os.path.join(log_dir, args.title + '_valid.csv')

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    # Load Datasets
    print('---> Loading Datasets...')
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    
    valid_transform = transforms.Compose([
        # transforms.CenterCrop((32, 32)),  
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    training_data = torchvision.datasets.CIFAR10(root=args.train_data_dir, train=True, download=False,
                                                 transform=train_transform)  
    valid_date = torchvision.datasets.CIFAR10(root=args.valid_data_dir, train=False, download=False,
                                              transform=valid_transform)  

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_date, batch_size=args.batch_size, shuffle=False,drop_last=True)

    
    # Load Models
    print('---> Constructing Network Architectures...')
    color_band = 1 if args.gray else 3   
    encoder = Encoder(args.data_depth, args.hidden_size, color_band) 
    decoder = Decoder(args.data_depth, args.hidden_size, color_band)  
    discriminator = ResNet(ResidualBlock)      
    discriminator.load_state_dict(torch.load("./net_135_1.pth"))
    discriminator.eval()
    for p in discriminator.parameters():
        p.requires_grad = False

    # VGG for perceptual loss   
    print('---> Constructing VGG-16 for Perceptual Loss...')
    vgg = utils2.VGGLoss(3,1,False)  

    # breakpoint()
    # Define Loss    
    print('---> Defining Loss...')

    optimizer_coders = optim.Adam(encoder.parameters(),
                                  lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer_coders, step_size=10, gamma=0.1)  

    mse_loss = torch.nn.MSELoss()   
    bce_loss = torch.nn.BCELoss()  


    if args.command == 'continue':
        encoder.load_state_dict(continue_from_checkpoint['encoder_state_dict'])
        decoder.load_state_dict(continue_from_checkpoint['decoder_state_dict'])
        discriminator.load_state_dict(continue_from_checkpoint['discriminator_state_dict'])
        scheduler.load_state_dict(continue_from_checkpoint['scheduler_state_dict'])

    # Use GPU
    if args.cuda:
        print('---> Loading into GPU memory...')
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        mse_loss.cuda()
        bce_loss.cuda()
        vgg.cuda()


    start_epoch = 0
    iteration = 0
    if args.command == 'continue':
        start_epoch = continue_from_checkpoint['epoch'] + 1
        iteration = continue_from_checkpoint['iteration']

    metric_names = ['adv_loss','mse_loss', 'vgg_loss', 'decoder_loss', 'loss',
                    'bit_err', 'decode_accuracy', 'psnr', 'ssim','acc']
    metrics = {m: 0 for m in metric_names}

    tic = time.time()
    for e in range(start_epoch, args.epochs):  
        print('---> Epoch %d starts training...' % e)
        epoch_start_time = time.time()
        # ------ train ------
        encoder.train()    

        discriminator.train() 
        i = 0  # batch idx
        train_iter = iter(train_loader)
        train_acc = 0
        valid_acc = 0
        while i < len(train_loader):   

            image_in, image_label = next(train_iter)  像
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()
                image_label = image_label.cuda()

            optimizer_coders.zero_grad()  
            stego = encoder(image_in, message_in)  

            g_on_stego = discriminator(stego)   

            ret, predictions = torch.max(g_on_stego.data, 1)

            correct_counts_stego = predictions.eq(image_label.data.view_as(predictions))
            acc = torch.mean(correct_counts_stego.type(torch.FloatTensor))
            pred_probs = F.softmax(g_on_stego, dim=1)
            onehot_labels = torch.eye(10, device=device)[image_label]
            real = torch.sum(onehot_labels * pred_probs, dim=1)  
            other, _ = torch.max((1 - onehot_labels) * pred_probs - (onehot_labels * 10000), dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            g_adv_loss = torch.sum(loss_adv)

            g_mse_loss = mse_loss(stego, image_in)  
            g_vgg_loss = torch.tensor(0.)      
            if args.use_vgg:
                vgg_on_cov = vgg(image_in)
                vgg_on_enc = vgg(stego)
                g_vgg_loss = mse_loss(vgg_on_enc, vgg_on_cov)

            g_decoder_loss = torch.tensor(0.)
            g_loss = lambda_adv * g_adv_loss + lambda_g * g_mse_loss  + lambda_p * g_vgg_loss
            g_loss.backward()   
            optimizer_coders.step()  

            with torch.no_grad():   

                decode_accuracy = torch.tensor(0.)
                bit_err = torch.tensor(0.)
                # ----------------------------------
                image_in = image_in * 255.0
                image_in = torch.round(image_in)
                stego = stego * 255.0
                stego_round = torch.round(stego)  

                metrics['adv_loss'] += g_adv_loss.item()
                metrics['mse_loss'] += g_mse_loss.item()
                metrics['vgg_loss'] += g_vgg_loss.item()
                metrics['decoder_loss'] += g_decoder_loss.item()
                metrics['loss'] += g_loss.item()
                metrics['decode_accuracy'] += decode_accuracy.item()
                metrics['bit_err'] += bit_err.item()
                metrics['psnr'] += utils2.psnr_between_batches(image_in, stego_round)
                metrics['ssim'] += pytorch_ssim.ssim(image_in/255.0, stego_round/255.0).item()
                metrics['acc'] += acc.item()

            i += 1
            iteration += 1


            if iteration % 50 == 0:
                for k in metrics.keys():
                    metrics[k] /= 50
                print('\nEpoch: %d, iteration: %d' % (e, iteration))
                for k in metrics.keys():
                    if 'loss' in k:
                        print(k + ': %.6f' % metrics[k], end='\t')
                print('')
                for k in metrics.keys():
                    if 'loss' not in k:
                        print(k + ': %.6f' % metrics[k], end='\t')
                utils2.write_losses(train_csv_file, iteration, e, metrics, time.time() - tic)
                for k in metrics.keys():
                    metrics[k] = 0


        val_metric_names = ['adv_loss','mse_loss', 'vgg_loss', 'decoder_loss', 'loss',
                        'bit_err', 'decode_accuracy', 'psnr', 'ssim', 'cover_acc','stego_acc']
        val_metrics = {m: 0 for m in val_metric_names}
        print('\n---> Epoch %d starts validating...' % e)
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        idx = 0
        file_names = list(i for i in range(0,1000))
        for batch_id, image in enumerate(valid_loader):
            image_in, image_label = image
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()
                image_label = image_label.cuda()
            with torch.no_grad():
                stego = encoder(image_in, message_in)  

                d_on_cover = discriminator(image_in)   
                d_on_stego = discriminator(stego)  


                ret, predictions = torch.max(d_on_cover.data, 1)
                correct_counts_cover = predictions.eq(image_label.data.view_as(predictions))
                acc_cover = torch.mean(correct_counts_cover.type(torch.FloatTensor))

                ret1, predictions1 = torch.max(d_on_stego.data, 1)
                correct_counts_stego = predictions1.eq(image_label.data.view_as(predictions1))
                acc_stego = torch.mean(correct_counts_stego.type(torch.FloatTensor))

                pred_probs = F.softmax(d_on_stego, dim=1)
                onehot_labels = torch.eye(10, device=device)[image_label]
                real = torch.sum(onehot_labels * pred_probs, dim=1)  
                other, _ = torch.max((1 - onehot_labels) * pred_probs - onehot_labels * 10000, dim=1)
                
                zeros = torch.zeros_like(other)
                loss_adv = torch.max(real - other, zeros)
                g_adv_loss = torch.sum(loss_adv)
                
                g_mse_loss = mse_loss(stego, image_in) 
                g_vgg_loss = torch.tensor(0.)     
                if args.use_vgg:
                    vgg_on_cov = vgg(image_in)
                    vgg_on_enc = vgg(stego)
                    g_vgg_loss = mse_loss(vgg_on_enc, vgg_on_cov)   

               
                g_decoder_loss = torch.tensor(0.)
                
                g_loss = lambda_adv * g_adv_loss + lambda_g * g_mse_loss + lambda_p * g_vgg_loss
                
                decode_accuracy = torch.tensor(0.)
                bit_err = torch.tensor(0.)
               
                image_in = image_in * 255.0
                image_in = torch.round(image_in)
                stego_round = stego * 255.0
                stego_round = torch.round(stego_round)
                
                val_metrics['adv_loss'] += g_adv_loss.item()
                val_metrics['mse_loss'] += g_mse_loss.item()
                val_metrics['vgg_loss'] += g_vgg_loss.item()
                val_metrics['decoder_loss'] += g_decoder_loss.item()
                val_metrics['loss'] += g_loss.item()
                val_metrics['decode_accuracy'] += decode_accuracy.item()
                val_metrics['bit_err'] += bit_err.item()
                val_metrics['psnr'] += utils2.psnr_between_batches(image_in, stego_round)
                val_metrics['ssim'] += pytorch_ssim.ssim(image_in/255.0, stego_round/255.0).item()
                val_metrics['cover_acc'] += acc_cover
                val_metrics['stego_acc'] += acc_stego
            

            if e == args.epochs - 1:
                
                cover_path = args.cover_data_dir
                stego_path = args.stego_data_dir

                if not os.path.exists(stego_path):
                    os.makedirs(stego_path)

                if not os.path.exists(cover_path):
                    os.makedirs(cover_path)

                for idx_in_batch in range(batch_size):
                    if idx < 1000:
                        cover_file_path = os.path.join(cover_path, str(file_names[idx]) + '.jpg')
                        stego_file_path = os.path.join(stego_path, str(file_names[idx]) + '.jpg')

                        cover_image = image_in[idx_in_batch].clamp(0.0, 255.0).permute(1, 2, 0)
                        stego_image = stego_round[idx_in_batch].clamp(0.0, 255.0).permute(1, 2, 0)

                        cover_image = cover_image.detach().cpu().numpy().astype('uint8')
                        stego_image = stego_image.detach().cpu().numpy().astype('uint8')
                        io.imsave(cover_file_path, cover_image)
                        io.imsave(stego_file_path, stego_image)
                        idx += 1
                    else:
                        break
        
        for k in val_metrics.keys():
            val_metrics[k] /= len(valid_loader)   
        print('Valid epoch: {}'.format(e))
        for k in val_metrics.keys():
            if 'loss' in k:
                print(k + ': %.6f' % val_metrics[k], end='\t')
        print('')
        for k in val_metrics.keys():
            if 'loss' not in k:
                print(k + ': %.6f' % val_metrics[k], end='\t')
        print('time:%.0f' % (time.time() - tic))
        print('Epoch %d finished, taking %0.f seconds\n' % (e, time.time() - epoch_start_time))
        utils2.write_losses(valid_csv_file, iteration, e, val_metrics, time.time() - tic)

        scheduler.step()

        # save model
        if (e + 1) % 10 == 0 or e == args.epochs - 1:
            checkpoint = {
                'epoch': e,
                'iteration': iteration,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_coders_state_dict': optimizer_coders.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics
            }
            filename = os.path.join(checkpoints_dir, "epoch_" + str(e) + ".pt")
            torch.save(checkpoint, filename)




if __name__ == '__main__':
    main()
