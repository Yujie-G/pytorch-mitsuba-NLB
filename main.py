import os
import sys

import torch

import model
from utils import *
from compress import compress_mat
from visualize import merl_sample

if __name__ == '__main__':
    assert len(sys.argv) == 3

    mat1 = sys.argv[1]
    mat2 = sys.argv[2]

    latent1 = compress_mat(mat1)
    latent2 = compress_mat(mat2)

    # latent1 = torch.randn(96).cuda().reshape(1, 1, -1)
    # latent2 = torch.randn(96).cuda().reshape(1, 1, -1)
    #############################################################################
    config = LayerDecoderOnly_alsTN_Config()

    layerer = getattr(model, config.layer_model)(config)
    layerer.load_state_dict(torch.load(config.layerer_path)())
    layerer = layerer.cuda().eval()

    # ''' albedo (R, G, B) | sigmaT ''''
    RGBS = torch.tensor([1.0, 1.0, 1.0, 0.0]).cuda().reshape(1, 1, -1)
    latent = torch.cat([latent1, latent2, RGBS], axis=-1).reshape(-1)


    layer_input1 = torch.cat([
        latent[0:32], latent[96: 96 + 32], latent[-4:-3] + 1, latent[-1:] + 1  # ! norm
    ])
    layer_input2 = torch.cat([
        latent[32:64], latent[96 + 32: 96 + 64], latent[-3:-2] + 1, latent[-1:] + 1
    ])
    layer_input3 = torch.cat([
        latent[64:96], latent[96 + 64: 96 + 96], latent[-2:-1] + 1, latent[-1:] + 1
    ])

    layer_output1 = layerer(layer_input1 - 1).detach().reshape(1, 1, 32) + 1
    layer_output2 = layerer(layer_input2 - 1).detach().reshape(1, 1, 32) + 1
    layer_output3 = layerer(layer_input3 - 1).detach().reshape(1, 1, 32) + 1

    output_latent = torch.cat([layer_output1, layer_output2, layer_output3], axis=-1).reshape(-1)
    print(output_latent)
    #############################################################################
    config = DecoderOnlyConfig()
    decoder = getattr(model, config.compress_decoder)(config)
    decoder.load_state_dict(torch.load(config.decoder_path)())
    decoder = decoder.cuda()
    merl_sample(decoder, output_latent, "/lizixuan/layeredbsdf-master/render-scene-and-result/matpreview/merged.merl")
    os.system(
        'source /lizixuan/layeredbsdf-master/setpath.sh && mitsuba /lizixuan/layeredbsdf-master/render-scene-and-result/matpreview/matpreview.xml')

