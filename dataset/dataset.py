import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string
import cv2
import os
import re
sys.path.append('../')
from utils import str_filt
from utils import utils_deblur
from utils import utils_sisr as sr
import imgaug.augmenters as iaa
from scipy import io as sio
from setup import CharsetMapper
scale = 0.90
kernel = utils_deblur.fspecial('gaussian', 15, 1.)
noise_level_img = 0.
class Lable2Tensor:
    def __init__(self):
        self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.english_dict = {}
        for index in range(len(self.english_alphabet)):
            self.english_dict[self.english_alphabet[index]] = index
        self.charset = CharsetMapper('./dataset/charset_36.txt', max_length=26)

    def label_encoder(self, label):
        batch = len(label)
        length = [len(i) for i in label]
        length_tensor = torch.Tensor(length).long().cuda()

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = torch.Tensor(text_gt).long().cuda()

        input_tensor = torch.from_numpy(input_tensor).long().cuda()
        return length_tensor, input_tensor, text_gt

    def getlabletensor(self, label):
        label_id = [torch.tensor(self.charset.get_labels(l.lower())) for l in label]
        label_id = torch.stack(label_id).cuda()
        label = [str_filt(i, 'lower') + '-' for i in label]  # to lower-case
        length_tensor, input_tensor, text_gt = self.label_encoder(label)
        return label_id, length_tensor, input_tensor, text_gt



def rand_crop(im):
    w, h = im.size
    p1 = (random.uniform(0, w*(1-scale)), random.uniform(0, h*(1-scale)))
    p2 = (p1[0] + scale*w, p1[1] + scale*h)
    return im.crop(p1 + p2)


def central_crop(im):
    w, h = im.size
    p1 = (((1-scale)*w/2), (1-scale)*h/2)
    p2 = ((1+scale)*w/2, (1+scale)*h/2)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im



class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=31, test=True):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        self.max_len = max_len
        self.voc_type = voc_type

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())

        try:
            img = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')

        label_str = str_filt(word, self.voc_type)
        return img, label_str


def add_shot_gauss_noise(rgb, shot_noise_mean, read_noise):
    noise_var_map = shot_noise_mean * rgb + read_noise
    noise_dev_map = np.sqrt(noise_var_map)
    noise = np.random.normal(loc=0.0, scale = noise_dev_map, size=None)
    if (rgb.mean() > 252.0):
        noise_rgb = rgb
    else:
        noise_rgb = rgb + noise
    noise_rgb = np.clip(noise_rgb, 0.0, 255.0)
    return noise_rgb



def gauss_unsharp_mask(rgb, shp_kernel, shp_sigma, shp_gain):
    LF = cv2.GaussianBlur(rgb, (shp_kernel, shp_kernel), shp_sigma)
    HF = rgb - LF
    RGB_peak = rgb + HF * shp_gain
    RGB_noise_NR_shp = np.clip(RGB_peak, 0.0, 255.0)
    return RGB_noise_NR_shp, LF



def degradation(src_img):
    # RGB Image input
    GT_RGB = np.array(src_img)
    GT_RGB = GT_RGB.astype(np.float32)

    pre_blur_kernel_set = [3, 5]
    sharp_kernel_set = [3, 5]
    blur_kernel_set = [5, 7, 9, 11]
    NR_kernel_set = [3, 5]

    # Pre Blur
    kernel = pre_blur_kernel_set[random.randint(0, (len(pre_blur_kernel_set) - 1))]
    blur_sigma = random.uniform(5., 6.)
    RGB_pre_blur = cv2.GaussianBlur(GT_RGB, (kernel, kernel), blur_sigma)

    rand_p = random.random()
    if rand_p > 0.2:
        # Noise
        shot_noise = random.uniform(0, 0.005)
        read_noise = random.uniform(0, 0.015)
        GT_RGB_noise = add_shot_gauss_noise(RGB_pre_blur, shot_noise, read_noise)
    else:
        GT_RGB_noise = RGB_pre_blur

    # Noise Reduction
    choice = random.uniform(0, 1.0)
    GT_RGB_noise = np.round(GT_RGB_noise)
    GT_RGB_noise = GT_RGB_noise.astype(np.uint8)
    # if (shot_noise < 0.06):
    if (choice < 0.7):
        NR_kernel = NR_kernel_set[random.randint(0, (len(NR_kernel_set) - 1))]  ###3,5,7,9
        NR_sigma = random.uniform(2., 3.)
        GT_RGB_noise_NR = cv2.GaussianBlur(GT_RGB_noise, (NR_kernel, NR_kernel), NR_sigma)
    else:
        value_sigma = random.uniform(70, 80)
        space_sigma = random.uniform(70, 80)
        GT_RGB_noise_NR = cv2.bilateralFilter(GT_RGB_noise, 7, value_sigma, space_sigma)

    # Sharpening
    GT_RGB_noise_NR = GT_RGB_noise_NR.astype(np.float32)
    shp_kernel = sharp_kernel_set[random.randint(0, (len(sharp_kernel_set) - 1))]  ###5,7,9
    shp_sigma = random.uniform(2., 3.)
    shp_gain = random.uniform(3., 4.)
    RGB_noise_NR_shp, LF = gauss_unsharp_mask(GT_RGB_noise_NR, shp_kernel, shp_sigma, shp_gain)

    # print("RGB_noise_NR_shp:", RGB_noise_NR_shp.shape)

    return Image.fromarray(RGB_noise_NR_shp.astype(np.uint8))


def noisy(noise_typ,image):

    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 50
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        # print("gauss:", np.unique(gauss))
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def JPEG_compress(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    ret_img = cv2.imdecode(encimg, 1)
    return ret_img


class lmdbDataset_real(Dataset):
    def __init__(
                 self, root=None,
                 voc_type='upper',
                 max_len=100,
                 test=False,
                 cutblur=False,
                 manmade_degrade=False,
                 rotate=None
                 ):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.cb_flag = cutblur
        self.rotate = rotate

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("nSamples:", nSamples)
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        self.manmade_degrade = manmade_degrade

    def __len__(self):
        return self.nSamples

    def rotate_img(self, image, angle):
        if not angle == 0.0:
            image = np.array(image)
            (h, w) = image.shape[:2]
            scale = 1.0
            # set the rotation center
            center = (w / 2, h / 2)
            # anti-clockwise angle in the function
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, M, (w, h))
            # back to PIL image
            image = Image.fromarray(image)
            
        return image


    def cutblur(self, img_hr, img_lr):
        p = random.random()

        img_hr_np = np.array(img_hr)
        img_lr_np = np.array(img_lr)

        randx = int(img_hr_np.shape[1] * (0.2 + 0.8 * random.random()))

        if p > 0.7:
            left_mix = random.random()
            if left_mix <= 0.5:
                img_lr_np[:, randx:] = img_hr_np[:, randx:]
            else:
                img_lr_np[:, :randx] = img_hr_np[:, :randx]

        return Image.fromarray(img_lr_np)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = ""# str(txn.get(label_key).decode())
        # print("in dataset....")
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            if self.manmade_degrade:
                img_lr = degradation(img_HR)
            else:
                img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            # print("GOGOOGO..............", img_HR.size)
            if self.cb_flag and not self.test:
                img_lr = self.cutblur(img_HR, img_lr)

            if not self.rotate is None:

                if not self.test:
                    angle = random.random() * self.rotate * 2 - self.rotate
                else:
                    angle = 0 #self.rotate

                # img_HR = self.rotate_img(img_HR, angle)
                # img_lr = self.rotate_img(img_lr, angle)

            img_lr_np = np.array(img_lr).astype(np.uint8)
            img_lry = cv2.cvtColor(img_lr_np, cv2.COLOR_RGB2YUV)
            img_lry = Image.fromarray(img_lry)

            img_HR_np = np.array(img_HR).astype(np.uint8)
            img_HRy = cv2.cvtColor(img_HR_np, cv2.COLOR_RGB2YUV)
            img_HRy = Image.fromarray(img_HRy)
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())
            # print("img_HR:", img_HR.size, img_lr.size())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, img_HRy, img_lry, label_str


class lmdbDataset_realIC15(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False, rotate=None):
        super(lmdbDataset_realIC15, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.degrade = True

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test
        print("We have", self.nSamples, "samples from", root)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        index = index % (self.nSamples+1)

        # print(self.nSamples, index)

        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_key = b'image-%09d' % index  # 128*32
        # img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_key, 'RGB')

            img_lr = img_HR

            img_lr_np = np.array(img_lr).astype(np.uint8)
            # print("img_lr_np:", img_lr_np.shape)

            if img_lr_np.shape[0] * img_lr_np.shape[1] > 1024:
                return self[(index + 1) % self.nSamples]
            if self.degrade:
                img_lr = degradation(img_lr)
            if img_lr.size[0] < 4 or img_lr.size[1] < 4:
                return self[index + 1]
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, img_HR, img_lr, label_str

class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug=None, blur=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
        self.aug = aug

        self.blur = blur

    def __call__(self, img, ratio_keep=False):

        size = self.size

        if ratio_keep:
            ori_width, ori_height = img.size
            ratio = float(ori_width) / ori_height

            if ratio < 3:
                width = 100# if self.size[0] == 32 else 50
            else:
                width = int(ratio * self.size[1])

            size = (width, self.size[1])

        # print("size:", size)
        img = img.resize(size, self.interpolation)

        if self.blur:
            # img_np = np.array(img)
            # img_np = cv2.GaussianBlur(img_np, (5, 5), 1)
            #print("in degrade:", np.unique(img_np))
            # img_np = noisy("gauss", img_np).astype(np.uint8)
            # img_np = apply_brightness_contrast(img_np, 40, 40).astype(np.uint8)
            # img_np = JPEG_compress(img_np)

            # img = Image.fromarray(img_np)
            pass

        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])

        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class NormalizeOnly(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug=None, blur=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
        self.aug = aug

        self.blur = blur

    def __call__(self, img, ratio_keep=False):

        size = self.size

        if ratio_keep:
            ori_width, ori_height = img.size
            ratio = float(ori_width) / ori_height

            if ratio < 3:
                width = 100# if self.size[0] == 32 else 50
            else:
                width = int(ratio * self.size[1])
            size = (width, self.size[1])


        if self.blur:
            img_np = np.array(img)
            img = Image.fromarray(img_np)

        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])

        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor



class resizeNormalizeRandomCrop(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img, interval=None):

        w, h = img.size

        if w < 32 or not interval is None:
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)
        else:
            np_img = np.array(img)
            h, w = np_img.shape[:2]
            np_img_crop = np_img[:, int(w * interval[0]):int(w * interval[1])]
            # print("size:", self.size, np_img_crop.shape, np_img.shape, interval)
            img = Image.fromarray(np_img_crop)
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)

        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor




class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples



class alignCollate_syn(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True,
                 y_domain=False
                 ):

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)

        aug = [
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(1, 5)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(
                d=(3, 9), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.MotionBlur(k=3),
            iaa.MeanShiftBlur(),
            iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(1, 7))
        ]

        self.aug = iaa.Sequential([sometimes(a) for a in aug], random_order=True)

        # self.y_domain = y_domain

        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        # self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.alphabet = open("al_chinese.txt", "r",encoding='UTF-8').readlines()[0].replace("\n", "")
        self.d2a = "-" + self.alphabet
        self.alsize = len(self.d2a)
        self.a2d = {}
        cnt = 0
        for ch in self.d2a:
            self.a2d[ch] = cnt
            cnt += 1

        imgH = self.imgH
        imgW = self.imgW

        self.transform = resizeNormalize((imgW, imgH), self.mask)
        self.transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask, blur=True)
        self.transform_pseudoLR = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask, aug=self.aug)

        self.train = train

    def degradation(self, img_L):
        img_L = np.array(img_L)
        img_L = sr.srmd_degradation(img_L, kernel)
        noise_level_img = 0.
        if not self.train:
            np.random.seed(seed=0)  # for reproducibility
        img_L = img_L + np.random.normal(0, noise_level_img, img_L.shape)

        return Image.fromarray(img_L.astype(np.uint8))

    def __call__(self, batch):
        images, images_lr, _, _, label_strs = zip(*batch)
        # images_hr = [self.degradation(image) for image in images]
        images_hr = images

        images_hr = [self.transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        if self.train:
            images_lr = [image.resize(
            (image.size[0] // 2, image.size[1] // 2), # self.down_sample_scale
            Image.BICUBIC) for image in images_lr]
        else:
            pass
        #    # for image in images_lr:
        #    #     print("images_lr:", image.size)
        #    images_lr = [image.resize(
        #         (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),  # self.down_sample_scale
        #        Image.BICUBIC) for image in images_lr]
        #    pass
        # images_lr = [self.degradation(image) for image in images]
        images_lr = [self.transform2(image) for image in images_lr]

        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        max_len = 26

        label_batches = []
        weighted_tics = []
        weighted_masks = []
        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            if len(label_list) <= 0:
                # blank label
                weighted_masks.append(0)
            else:
                weighted_masks.extend(label_list)

            labels = torch.tensor(label_list)[:, None].long()

            if labels.shape[0] > 0:
                label_vecs = torch.zeros((labels.shape[0], self.alsize))
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
                weighted_tics.append(1)
            else:
                label_vecs = torch.zeros((1, self.alsize))
                label_vecs[0, 0] = 1.
                label_batches.append(label_vecs)
                weighted_tics.append(0)

        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        # print(images_lr.shape, images_hr.shape)

        return images_hr, images_lr, images_hr, images_lr, label_strs, label_rebatches, torch.tensor(weighted_masks).long(), torch.tensor(weighted_tics)



class alignCollate_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, images_HRy, images_lry, label_strs = zip(*batch)

        new_images_HR = []
        new_images_LR = []
        new_label_strs = []
        if type(images_HR[0]) == list:
            for image_item in images_HR:
                new_images_HR.extend(image_item)

            for image_item in images_lr:
                new_images_LR.extend(image_item)

            for image_item in label_strs:
                new_label_strs.extend(image_item)

            images_HR = new_images_HR
            images_lr = new_images_LR
            label_strs = new_label_strs

        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_HR, images_lr, label_strs


class alignCollate_realWTL(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, images_HRy, images_lry, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        images_lry = [self.transform2(image) for image in images_lry]
        images_lry = torch.cat([t.unsqueeze(0) for t in images_lry], 0)

        images_HRy = [self.transform(image) for image in images_HRy]
        images_HRy = torch.cat([t.unsqueeze(0) for t in images_HRy], 0)

        max_len = 26

        label_batches = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                padding = int(inter_com / (len(word) - 1))
                new_word = word[0]
                for i in range(len(word) - 1):
                   new_word += "-" * padding + word[i+1]

                word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            if labels.shape[0] > 0:
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
            else:
                label_batches.append(label_vecs)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        return images_HR, images_lr, images_HRy, images_lry, label_strs, label_rebatches


class alignCollate_realWTLAMask(alignCollate_syn):

    def get_mask(self, image):
        img_hr = np.transpose(image.data.numpy() * 255, (1, 2, 0))
        img_hr_gray = cv2.cvtColor(img_hr[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # print("img_hr_gray: ", np.unique(img_hr_gray), img_hr_gray.shape)
        kernel = np.ones((5, 5), np.uint8)
        hr_canny = cv2.Canny(img_hr_gray, 20, 150)
        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        hr_canny = cv2.GaussianBlur(hr_canny, (5, 5), 1)
        weighted_mask = 0.4 + (hr_canny / 255.0) * 0.6
        return torch.tensor(weighted_mask).float().unsqueeze(0)

    def __call__(self, batch):
        images_HR, images_lr, images_HRy, images_lry, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        # images_pseudoLR = [self.transform2(image) for image in images_HR]
        # images_pseudoLR = torch.cat([t.unsqueeze(0) for t in images_pseudoLR], 0)
        # align
        images_pseudoLR = None
        # resize + expend - 4
        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        images_lry = [self.transform2(image) for image in images_lry]
        images_lry = torch.cat([t.unsqueeze(0) for t in images_lry], 0)

        images_HRy = [self.transform(image) for image in images_HRy]
        images_HRy = torch.cat([t.unsqueeze(0) for t in images_HRy], 0)

        # print("images_lry:", images_lry.shape)

        # weighted_masks = [self.get_mask(image_HR) for image_HR in images_HR]
        # weighted_masks = torch.cat([t.unsqueeze(0) for t in weighted_masks], 0)

        # print("weighted_masks:", weighted_masks.shape, np.unique(weighted_masks))
        max_len = 26

        label_batches = []
        weighted_masks = []
        weighted_tics = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                # word = [ch for ch in word]
                # word[2] = "e"
                # word = "".join(word)
                pass
            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                padding = int(inter_com / (len(word) - 1))
                new_word = word[0]
                for i in range(len(word) - 1):
                    new_word += "-" * padding + word[i+1]

                word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            #########################################
            # random.shuffle(label_list)
            #########################################
            
            if len(label_list) <= 0:
                # blank label
                weighted_masks.append(0)
            else:
                weighted_masks.extend(label_list)

            # word_len = len(word)
            # if word_len > max_len:
            #     max_len = word_len
            # print("label_list:", word, label_list)
            labels = torch.tensor(label_list)[:, None].long()

            # print("labels:", labels)

            if labels.shape[0] > 0:
                label_vecs = torch.zeros((labels.shape[0], self.alsize))
                # print(label_vecs.scatter_(-1, labels, 1))
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
                weighted_tics.append(1)
            else:
                label_vecs = torch.zeros((1, self.alsize))
                # Assign a blank label
                label_vecs[0, 0] = 1.
                label_batches.append(label_vecs)
                weighted_tics.append(0)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        return images_HR, images_pseudoLR, images_lr, images_HRy, images_lry, label_strs, label_rebatches, torch.tensor(weighted_masks).long(), torch.tensor(weighted_tics)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
