import cv2
import os
import numpy as np
from PIL import Image
from .system import Reconstructor
from typing import Tuple
import torch


def resize_colorized(y_color, y_gr):
    y_color = cv2.resize(y_color, (y_gr.shape[1], y_gr.shape[0]))
    y_lab = cv2.cvtColor(y_color, cv2.COLOR_RGB2LAB)
    L = y_gr[:, :, 0][:, :, np.newaxis]
    AB = y_lab[:, :, 1:]
    LAB = np.concatenate((L, AB), axis=2)
    res = cv2.cvtColor(LAB, cv2.COLOR_LAB2RGB)
    return res


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_predictions(image, pred, save_path, name):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    N_variants = pred.shape[0]
    for i in range(N_variants):
        imgc = pred[i]
        imgc = np.moveaxis(imgc, 0, 2).astype('uint8')
        imgc = resize_colorized(imgc, gray)
        imgc = Image.fromarray(imgc)
        save_name = os.path.join(save_path, name.replace('.', '_' + str(i) + '.'))
        imgc.save(save_name)


class Artist:
    def __init__(self, model_checkpoint, device=torch.device('cuda:0')):
        model = Reconstructor(model_checkpoint)
        self.device = device
        self.model = model.to(device)

    def preprocess_image(self, image):
        img = cv2.resize(image, self.model.params['size'], interpolation=3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        gray = np.moveaxis(np.array(gray), 2, 0)
        gray = torch.tensor(gray / (255.0 * 0.5) - 1, dtype=torch.float).unsqueeze(0)
        return gray

    def colorize(
            self,
            image_path: str,
            num_variants: int
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray]:

        original_image = read_image(image_path)
        grayscale_image = self.preprocess_image(original_image)
        grayscale_image = torch.cat([grayscale_image] * num_variants, dim=0)

        prediction, prediction_step = self.model.restoration(grayscale_image.to(self.device), sample_num=4)
        prediction = (prediction + 1) * 255 / 2
        prediction_step = (prediction_step + 1) * 255 / 2
        return original_image, prediction.cpu().numpy(), prediction_step.cpu().numpy()

    @staticmethod
    def visualize(image, pred, offset=5, original_size=True, inline=3):

        num_variants = pred.shape[0]
        if not original_size:
            width = 1024
            height = int(image.shape[0] / image.shape[1] * width)
            image = cv2.resize(image, (width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        images = [Image.fromarray(image)]
        for i in range(num_variants):
            imgc = pred[i]
            imgc = np.moveaxis(imgc, 0, 2).astype('uint8')
            imgc = resize_colorized(imgc, gray)
            images.append(Image.fromarray(imgc))

        chunks = [images[x:x + inline] for x in range(0, len(images), inline)]

        im_line = []
        for line in chunks:
            im_line.append(concat_v(line, offset=offset))
        new_im = concat_h(im_line, offset=offset)

        return new_im


def concat_v(images, offset=5):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + offset * (len(images) - 1)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + offset
    return new_im


def concat_h(images, offset=5):
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights) + offset * (len(images) - 1)
    max_width = max(widths)
    new_im = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + offset
    return new_im
