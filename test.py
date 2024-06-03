import os
import sys
from tqdm.auto import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from datasets.gastrointestinal import Gastrointestinal, RandomGenerator
from pathlib import Path
from torchvision import transforms

from icecream import ic


# class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}
class_to_name = {1: 'bleeding'}

def inference(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['dataset'](root_path=args.test_root_path,
                                   anno_path=Path(args.valid_root_path) / '_annotations.coco.json',
                                   split='test',
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size],
                                                        low_res=[512, 512],
                                                        split='valid')]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes + 1):
        try:
            logging.info(f'Mean class {i} name {class_to_name[i]} mean_dice {metric_list[i - 1][0]} mean_hd95 {metric_list[i - 1][1]}')
        except:
            logging.info(f'Mean class {i} mean_dice {metric_list[i - 1][0]} mean_hd95 {metric_list[i - 1][1]}')
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(f'Testing performance in best val model: mean_dice : {performance} mean_hd95 : {mean_hd95}')
    logging.info("Testing Finished!")
    return 1


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def test_one_epoch(model, valid_dataloader):
    loss_list, loss_ce_list, loss_dice_list = [], [], []
    for i_batch, sampled_batch in tqdm(enumerate(valid_dataloader)):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        low_res_label_batch = low_res_label_batch.cuda()
        assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

        outputs = model(image_batch, multimask_output, args.img_size)
        loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, CrossEntropyLoss(), DiceLoss(args.num_classes + 1), 0.8)

        logging.info(f'[Test phase] loss: {loss.item():.4f}, loss_ce: {loss_ce.item():.4f}, loss_dice: {loss_dice.item():.4f}')
        loss_list.append(loss.item())
        loss_ce_list.append(loss_ce.item())
        loss_dice_list.append(loss_dice.item())
    logging.info(f'[Test phase] Mean loss: {sum(loss_list)/len(loss_list):.4f}, mean loss_ce: {sum(loss_ce_list)/len(loss_ce_list):.4f}, mean loss_dice: {sum(loss_dice_list)/len(loss_dice_list):.4f}')

def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--valid_root_path', type=str, default=r'data\map_1-1_gastrointestinal_coco\valid')
    parser.add_argument('--dataset', type=str, default='gastrointestinal', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', 
                        type=str, 
                        default=r'output\MedSAM\results\gastrointestinal_512_pretrain_vit_b_epo200_bs5_lr0.0001\testing')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default=r'MedSAM\medsam_vit_b.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default=r'output\MedSAM\results\gastrointestinal_512_pretrain_vit_b_epo200_bs5_lr0.0001\best_epoch=29_valid_loss=0.088.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'gastrointestinal': {
            'dataset': Gastrointestinal,
            'root_path': args.valid_root_path,
            'num_classes': args.num_classes,
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    # inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)

    db_test = Gastrointestinal(root_path=args.valid_root_path,
                                   anno_path=Path(args.valid_root_path) / '_annotations.coco.json',
                                   split='test',
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size],
                                                        low_res=[128, 128],
                                                        split='valid')]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    test_one_epoch(model=net,
                   valid_dataloader=testloader)
