import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from torchvision import transforms
from PIL import Image
from segment_anything import sam_model_registry as sam_model_registry_lora
from pathlib import Path
from importlib import import_module
from tqdm.auto import tqdm


def init_model(checkpoint_sam='SAMed/MedSAM/medsam_vit_b.pth',
               checkpoint_lora='SAMed/gastrointestinal_512_pretrain_vit_b_epo200_bs5_lr0.0001/best_epoch=102_valid_loss=0.068.pth',
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    sam_model, _ = sam_model_registry_lora['vit_b'](image_size=512, 
                                                num_classes=2,
                                                checkpoint=checkpoint_sam, 
                                                pixel_mean=[0, 0, 0],
                                                pixel_std=[1, 1, 1])
    
    sam_model.to(device)

    pkg = import_module('sam_lora_image_encoder')
    net = pkg.LoRA_Sam(sam_model, 4).to(device)
    net.load_lora_parameters(checkpoint_lora)

    return net


def preprocess_image(img_path,
                     device):
    image_data = np.array(Image.open(str(img_path)))
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
    image_data_pre[image_data==0] = 0
    image_data_pre = np.uint8(image_data_pre)
    H, W, _ = image_data_pre.shape

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(512,512)),
                                    transforms.ToTensor()])
    img_tensor = transform(image_data_pre)
    img_tensor = img_tensor.unsqueeze(dim=0).to(device)


    return img_tensor, image_data, H, W

def inference(model,
              img_tensor,
              device):
    with torch.inference_mode():
        model.eval()
        
        output = model(img_tensor, False, 512)

    transform = transforms.Compose([transforms.Resize(size=(224,224))])

    output_masks = torch.argmax(torch.softmax(output['masks'], dim=1), dim=1, keepdim=True)
    output_masks = output_masks.squeeze(dim=0) * 50
    output_masks = transform(output_masks)
    output_masks_np_lora = output_masks.cpu().numpy()
    output_masks_np_lora = (output_masks_np_lora > 0).astype(np.uint8)

    return output_masks_np_lora

def show_mask(mask, ax, random_color=False):
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_img(image_data,
             output_masks_np_lora,
             save_path,
             save_name):
    _, axs = plt.subplots(1, 2, figsize=(25, 25))

    axs[0].imshow(image_data)
    axs[0].set_title('Original Image', fontsize=20)
    axs[0].axis('off')

    axs[1].imshow(image_data)
    show_mask(output_masks_np_lora, axs[1])
    axs[1].set_title('SAM LoRA', fontsize=20)
    axs[1].axis('off')

    plt.savefig(f'{save_path}/{save_name}')

    plt.clf()

    plt.close()



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path',
                        default='./results',
                        help='the path where visualize results stored')
    
    parser.add_argument('--img_path',
                        default='inference_pipeline/bleeding',
                        help='image folder path or single image path')

    return parser.parse_args()

if __name__ == '__main__':
    # set seeds
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model = init_model()

    for img_path in tqdm(Path(args.img_path).glob('*.png')):
        img_tensor, image_data, H, W = preprocess_image(img_path, device=device)
        
        output_masks_np_lora = inference(model,
                                         img_tensor=img_tensor,
                                         device=device)
        
        save_img(image_data,
                 output_masks_np_lora,
                 save_path,
                 img_path.name)