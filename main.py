from model.inference import Artist, visualize
import torch
import os
import random

if __name__ == "__main__":
    model_cp = 'v0.cp'
    device = torch.device('cuda:2')
    artist = Artist(model_checkpoint=model_cp, device=device)

    image_path = 'artem.jpg'
    image, prediction, pred_step = artist.colorize(image_path, 5)
    im_res = visualize(image, prediction,original_size=False, inline=3)
    im_res.save("within_cont.png")

    print(torch.cuda.get_device_name("cuda:0"))
    print(torch.cuda.get_device_name("cuda:1"))
    print(torch.cuda.get_device_name("cuda:2"))
