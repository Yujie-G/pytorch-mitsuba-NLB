'''
    This is the experimental code for paper ``Fan et al., Neural Layered BRDFs, SIGGRAPH 2022``
    
    This script is suboptimal and experimental. 
    There may be redundant lines and functionalities. 
    This code is provided on an ''AS IS'' basis WITHOUT WARRANTY of any kind. 
    One can arbitrarily change or redistribute these scripts with above statements.
    	
    Jiahui Fan, 2022/09
'''


'''
    This script helps visualize any latent code into outgoing radiance distributions.
'''

import sys

import torch
import warnings
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
import numpy as np
import math
from tqdm import tqdm

import exr
from utils import *
import model
import coords
from merl import Merl


def cross_product(v1, v2):
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def normalize(v):
    len = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return [v[0] / len, v[1] / len, v[2] / len]

def rotate_vector(vector, axis, angle):
    cos_ang = math.cos(angle)
    sin_ang = math.sin(angle)
    dot_product = sum(axis[i] * vector[i] for i in range(3))
    temp = dot_product * (1 - cos_ang)

    cross = cross_product(axis, vector)

    return [
        vector[0] * cos_ang + axis[0] * temp + cross[0] * sin_ang,
        vector[1] * cos_ang + axis[1] * temp + cross[1] * sin_ang,
        vector[2] * cos_ang + axis[2] * temp + cross[2] * sin_ang
    ]

def half_angle_to_std(theta_h, theta_d, phi_d):
    wi_z = math.cos(theta_d)
    wi_xy = math.sin(theta_d)
    wi_x = wi_xy * math.cos(phi_d)
    wi_y = wi_xy * math.sin(phi_d)
    wi = [wi_x, wi_y, wi_z]
    wo = [-wi_x, -wi_y, wi_z]
    wi = normalize(wi)
    wo = normalize(wo)

    bi_normal = [0.0, 1.0, 0.0]
    normal = [0.0, 0.0, 1.0]

    wi_o = rotate_vector(wi, bi_normal, theta_h)
    wo_o = rotate_vector(wo, bi_normal, theta_h)

    return wi_o, wo_o

'''
    Outgoing radiance visualization, given an representation network and a latent code.

    Args:
        decoder: torch.nn.Module, the network to visualize latent codes
        latent: torch.tensor, latent code to visualize of size (latent_size,)
        out_anme: save file name (.exr)
        resolution (optional): the resolution of output image (Default: 512)
        wiz (optional): z-coordinate of the view direction (Default: 1.0, i.e., [0, 0, 1] view direction)
'''
def vis_ndf(decoder, latent, out_name, resolution=512, wiz=1):
        
    wix = np.sqrt(1 - wiz**2)
    wiy = 0
    with torch.no_grad():
        new_wiwo = []
        reso_step = 2.0 / resolution
        for wox in np.arange(-1.0, 1.0, reso_step):
            for woy in np.arange(-1.0, 1.0, reso_step):
                new_wiwo.append([wix, wiy, wox, woy])
        new_wiwo = torch.Tensor(np.array(new_wiwo)).unsqueeze(0)  # size: [1, resolution ** 2, 4]
        new_wiwo = new_wiwo.cuda()
        new_wiwo = to6d(new_wiwo)

        latent = latent.reshape(1, 1, 32*3).expand(
            1, resolution**2, 32*3)
        decoder_input = torch.cat([new_wiwo, latent[:, :, :32]], axis=-1) # size: [1, reso**2, 20]
        output0 = (decoder(decoder_input) * new_wiwo[:, :, -1:]).detach().cpu().numpy() # size: [1, n, 3]
        decoder_input = torch.cat([new_wiwo, latent[:, :, 32:-32]], axis=-1) # size: [1, reso**2, 20]
        output1 = (decoder(decoder_input) * new_wiwo[:, :, -1:]).detach().cpu().numpy() # size: [1, n, 3]
        decoder_input = torch.cat([new_wiwo, latent[:, :, -32:]], axis=-1) # size: [1, reso**2, 20]
        output2 = (decoder(decoder_input) * new_wiwo[:, :, -1:]).detach().cpu().numpy() # size: [1, n, 3]
        
        image = np.concatenate([output0, output1, output2], axis=0).reshape(3, resolution, resolution).transpose(1, 2, 0)

        ## drop out invalid points
        mid = resolution // 2
        for i in range(resolution):
            for j in range(resolution):
                distance = ((i - mid) / mid) ** 2 + ((j - mid) / mid) ** 2
                if distance > 1:
                    image[i, j, :] = 0.5

        exr.write32(image, out_name)


def merl_sample(decoder, latent, out_name, resolution=512, wiz=1):
    with torch.no_grad():
        # new_wiwo = []
        # for th in range(90):
        #     theta_h = th * th * math.pi/2
        #     for td in range(90):
        #         theta_d = td / 90.0 * math.pi/2
        #         for pd in range(180):
        #             phi_d = pd / 180.0 * math.pi
        #             wi, wo = half_angle_to_std(theta_h, theta_d, phi_d)
        #             wix = wi[0]
        #             wiy = wi[1]
        #             wox = wo[0]
        #             woy = wo[1]
        #             new_wiwo.append([wix, wiy, wox, woy])
        # new_wiwo = torch.Tensor(np.array(new_wiwo)).unsqueeze(0)
        # new_wiwo = new_wiwo.cuda()
        # new_wiwo = to6d(new_wiwo)

        res = torch.load('data/tabular_hd_all.pt').squeeze().to('cuda')
        new_wiwo = coords.hd_to_io(res[:, :3], res[:, 3:]) # new_wiwo (tuple): (wi(xyz), wo(xyz))
        new_wiwo = torch.cat([new_wiwo[0][:, :2], new_wiwo[1][:, :2]], dim=1).unsqueeze(0)
        new_wiwo = to6d(new_wiwo)
        # exr.write32(new_wiwo.reshape(1000, -1, 3).cpu().numpy(), 'wiwo.exr')

        # new_wiwo = torch.cat([new_wiwo[0], new_wiwo[1]], dim=1)[:, [0,1,3,4,2,5]].unsqueeze(0)



        latent = latent.reshape(1, 1, 32 * 3).expand(1, new_wiwo.shape[1], 32 * 3)
        batch_size = 1024*64  # Set your batch size
        num_batches = new_wiwo.shape[1] // batch_size
        output0_batches, output1_batches, output2_batches = [], [], []
        for i in tqdm(range(num_batches + 1)):
            start = i * batch_size
            end = min(start + batch_size, new_wiwo.shape[1])

            latent_batch = latent[:, start:end, :].reshape(1, -1, 32 * 3)
            new_wiwo_batch = new_wiwo[:, start:end, :]

            decoder_input = torch.cat([new_wiwo_batch, latent_batch[:, :, :32]], axis=-1)
            output0_batch = (decoder(decoder_input) * new_wiwo_batch[:, :, -1:]).detach().cpu().numpy()
            output0_batches.append(output0_batch)

            decoder_input = torch.cat([new_wiwo_batch, latent_batch[:, :, 32:-32]], axis=-1)
            output1_batch = (decoder(decoder_input) * new_wiwo_batch[:, :, -1:]).detach().cpu().numpy()
            output1_batches.append(output1_batch)

            decoder_input = torch.cat([new_wiwo_batch, latent_batch[:, :, -32:]], axis=-1)
            output2_batch = (decoder(decoder_input) * new_wiwo_batch[:, :, -1:]).detach().cpu().numpy()
            output2_batches.append(output2_batch)

        # Concatenate all batches
        output0 = np.concatenate(output0_batches, axis=1)
        output1 = np.concatenate(output1_batches, axis=1)
        output2 = np.concatenate(output2_batches, axis=1)

        pred = np.concatenate([output0, output1, output2], axis=0).reshape(3, -1).transpose(1, 0)

        # # normalize
        # mx = np.max(pred, axis=0)
        # mn = np.min(pred, axis=0)
        # pred = (pred - mn) / (mx - mn)
        # pred *= 1

        # exr.write(pred.reshape(90,-1,3), 'vis.exr')
        save_data = Merl()
        save_data.from_array(pred)
        save_data.write_merl_file(out_name)
        
        
if __name__ == '__main__':

    if len(sys.argv) > 3:
        print(
        '''
            Usage: $ python visualize.py [resolution] [wiz]
            
            Args:
                resolution (optional): the resolution of output image (Default: 512)
                wiz (optional): z-coordinate of the view direction (Default: 1.0, i.e., [0, 0, 1] view direction)
        '''
            )
        exit()
        
    config = DecoderOnlyConfig()
    decoder = getattr(model, config.compress_decoder)(config)
    decoder.load_state_dict(torch.load(config.decoder_path)())
    decoder = decoder.cuda()

    latent = torch.tensor([
        1.1987, 0.9763, 0.9509, 0.9067, 1.1457, 0.9931, 1.0734, 1.0939, 1.0885, 0.9078, 0.9283, 0.9823,
        0.9380, 1.0572, 1.0578, 1.0943, 0.9241, 1.0044, 1.0639, 1.0829, 1.1633, 0.9191, 1.0814, 0.9693,
        0.8199, 1.0564, 0.9662, 1.0435, 1.0290, 1.0040, 0.8947, 0.9505, 1.1898, 0.9695, 0.9544, 0.9108,
        1.1352, 0.9928, 1.0691, 1.0880, 1.0941, 0.9095, 0.9295, 0.9742, 0.9422, 1.0522, 1.0535, 1.0953,
        0.9156, 0.9975, 1.0680, 1.0784, 1.1551, 0.9212, 1.0752, 0.9658, 0.8253, 1.0494, 0.9734, 1.0364,
        1.0356, 0.9978, 0.8993, 0.9379, 1.1890, 0.9689, 0.9537, 0.9104, 1.1352, 0.9941, 1.0698, 1.0888,
        1.0948, 0.9087, 0.9285, 0.9736, 0.9409, 1.0531, 1.0540, 1.0950, 0.9160, 0.9987, 1.0691, 1.0794,
        1.1561, 0.9204, 1.0756, 0.9651, 0.8237, 1.0499, 0.9725, 1.0374, 1.0364, 0.9994, 0.8990, 0.9372,

    ]).cuda()
    resolution = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    wiz = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    # vis_ndf(decoder, latent, 'vis.exr', resolution=resolution, wiz=wiz)
    merl_sample(decoder, latent, "/lizixuan/layeredbsdf-master/render-scene-and-result/matpreview/merged.merl", resolution=resolution, wiz=wiz)
    os.system('source /lizixuan/layeredbsdf-master/setpath.sh && mitsuba /lizixuan/layeredbsdf-master/render-scene-and-result/matpreview/multilayered.xml')
