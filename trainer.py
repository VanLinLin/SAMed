import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic
from pathlib import Path


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


train_iter_num = 0
valid_iter_num = 0
def gastrointestinal(args, model, snapshot_path, multimask_output, low_res):
    from datasets.gastrointestinal import Gastrointestinal, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/gastrointestinal_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Gastrointestinal(root_path=args.train_root_path, 
                                anno_path=Path(args.train_root_path) / '_annotations.coco.json', 
                                split="train",
                                transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size], 
                                                     low_res=[low_res, low_res],
                                                     split='train')]))
    
    db_valid = Gastrointestinal(root_path=args.valid_root_path,
                                anno_path=Path(args.valid_root_path) / '_annotations.coco.json', 
                                split='valid',
                                transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size], 
                                                     low_res=[low_res, low_res],
                                                     split='valid')]))

    print(f"The length of train set is: {len(db_train)}")
    print(f"The length of valid set is: {len(db_valid)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, 
                             batch_size=batch_size, 
                             shuffle=True, 
                            #  num_workers=8, 
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    validloader = DataLoader(db_valid,
                             batch_size=batch_size,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/gastrointestinal_log')

    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch

    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations ")
    best_losses = np.inf
    
    def train_one_epoch(train_dataloader, max_iterations, writer, optimizer):
        global train_iter_num
        for i_batch, sampled_batch in enumerate(train_dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.warmup and train_iter_num < args.warmup_period:
                lr_ = base_lr * ((train_iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = train_iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = train_iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            train_iter_num = train_iter_num + 1
            writer.add_scalar('train/lr', lr_, train_iter_num)
            writer.add_scalar('train/total_loss', loss, train_iter_num)
            writer.add_scalar('train/loss_ce', loss_ce, train_iter_num)
            writer.add_scalar('train/loss_dice', loss_dice, train_iter_num)
            logging.info(f'[Train phase] Epoch: {epoch_num}, iteration: {train_iter_num}, loss: {loss.item():.4f}, loss_ce: {loss_ce.item():.4f}, loss_dice: {loss_dice.item():.4f}')

    def valid_one_epoch(valid_dataloader, writer):
        global valid_iter_num
        for i_batch, sampled_batch in enumerate(valid_dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)

            valid_iter_num = valid_iter_num + 1
            writer.add_scalar('valid/total_loss', loss, valid_iter_num)
            writer.add_scalar('valid/loss_ce', loss_ce, valid_iter_num)
            writer.add_scalar('valid/loss_dice', loss_dice, valid_iter_num)
            logging.info(f'[Valid phase] Epoch: {epoch_num}, iteration: {valid_iter_num}, loss: {loss.item():.4f}, loss_ce: {loss_ce.item():.4f}, loss_dice: {loss_dice.item():.4f}')

            if valid_iter_num % 20 == 0:
                image = image_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('valid/Image', image, valid_iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('valid/Prediction', output_masks[1, ...] * 50, valid_iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('valid/GroundTruth', labs, valid_iter_num)
        return loss


    for epoch_num in tqdm(range(max_epoch), ncols=70, desc='Training'):
        print()
        logging.info(f'{"*"*10}Start training{"*"*10}')
        train_one_epoch(train_dataloader=trainloader,
                        max_iterations=max_iterations,
                        writer=writer,
                        optimizer=optimizer)
        
        
        logging.info(f'{"*"*10}Start validing{"*"*10}')
        valid_loss = valid_one_epoch(valid_dataloader=validloader,
                                     writer=writer)


        save_interval = 20 # int(max_epoch/6)
        if epoch_num % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if valid_loss < best_losses:
            best_losses = valid_loss
            for previous_best_model in Path(snapshot_path).glob('best*.pth'):
                previous_best_model.unlink()

            save_mode_path = os.path.join(snapshot_path, f'best_epoch={epoch_num}_valid_loss={best_losses:.3f}.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)

            logging.info(f"save best model to {save_mode_path}")

        if epoch_num >= max_epoch or epoch_num >= stop_epoch:
            save_mode_path = os.path.join(snapshot_path, f'epoch={epoch_num}.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info(f"save model to {save_mode_path}")
            break

    writer.close()
    return "Training Finished!"