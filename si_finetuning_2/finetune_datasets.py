from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
import libs.clip
from PIL import Image
import clip
import os
import glob
import einops
import torchvision.transforms.functional as F
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn

"""
데이터 포맷 예시 
dataset = DialogueImageTextDataset(
    json_path="/home/ivpl-d29/dataset/AvaMERG/train_finetune.json",
    image_root="/home/ivpl-d29/dataset/AvaMERG/image_v5_0_64/"
)
"""
"""
image_root/
├── img001.png
├── img001_transformed.png
├── img002.png
├── img002_transformed.png
└── ...

json 파일은 하나의 리스트 

[
  {
    "input_image": "img001.png",
    "input_text": "이 강아지를 고양이처럼 바꿔줘.",
    "output_image": "img001_transformed.png",
    "output_text": "고양이처럼 생긴 모습이에요."
  },
  {
    "input_image": "img002.png",
    "input_text": "이 자동차를 미래형으로 바꿔줘.",
    "output_image": "img002_transformed.png",
    "output_text": "이건 미래 자동차처럼 보여요."
  }
]

"""

import random





class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10

class CIFAR10(DatasetFactory):
    r""" CIFAR10 dataset

    Information of the raw dataset:
         train: 50,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, path, random_flip=False, cfg=False, p_uncond=None):
        super().__init__()

        transform_train = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        transform_test = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        if random_flip:  # only for train
            transform_train.append(transforms.RandomHorizontalFlip())
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        self.train = datasets.CIFAR10(path, train=True, transform=transform_train, download=True)
        self.test = datasets.CIFAR10(path, train=False, transform=transform_test, download=True)

        assert len(self.train.targets) == 50000
        self.K = max(self.train.targets) + 1
        self.cnt = torch.tensor([len(np.where(np.array(self.train.targets) == k)[0]) for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt: {self.cnt}')
        print(f'frac: {self.frac}')

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)


    @property
    def data_shape(self):
        return 3, 32, 32

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_cifar10_train_pytorch.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# ImageNet


class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        z, label = np.load(path, allow_pickle=True)
        return z, label


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebA(DatasetFactory):
    r""" train: 162,770
         val:   19,867
         test:  19,962
         shape: 3 * width * width
    """

    def __init__(self, path, resolution=64):
        super().__init__()

        self.resolution = resolution

        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        transform = transforms.Compose([Crop(x1, x2, y1, y2), transforms.Resize(self.resolution),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        self.train = datasets.CelebA(root=path, split="train", target_type=[], transform=transform, download=True)
        self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz'

    @property
    def has_label(self):
        return False


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            prompt, context = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'





# ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
"""
class DialogueSubDataset(Dataset):
    def __init__(self, json_path, image_root, random_flip=False, p_uncond=None):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)

        self.image_root = image_root
        self.p_uncond = p_uncond

        transform = [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
        if random_flip:
            transform.insert(0, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_img = Image.open(os.path.join(self.image_root, item['input_image'])).convert("RGB")
        output_img = Image.open(os.path.join(self.image_root, item['output_image'])).convert("RGB")

        input_img = self.transform(input_img)
        output_img = self.transform(output_img)

        input_text = item['input_text']
        output_text = item['output_text']

        if self.p_uncond is not None and random.random() < self.p_uncond:
            input_text = ""

        return {
            "input_img": input_img,
            "input_text": input_text,
            "output_img": output_img,
            "output_text": output_text
        }


"""

from torch.utils.data import Dataset
import os
import json
import numpy as np
from PIL import Image
import einops


def center_crop(width, height, image):
    h, w, _ = image.shape
    top = max(0, (h - height) // 2)
    left = max(0, (w - width) // 2)
    return image[top:top+height, left:left+width]


class AvaMERGDatabase(Dataset):
    def __init__(self, json_path, image_root, size):
        self.image_root = image_root
        self.size = size
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        def load_and_process(image_name):
            path = os.path.join(self.image_root, image_name)
            image = Image.open(path).convert("RGB")
            # 해상도 리사이즈 추가 (self.size x self.size)
            image = image.resize((self.size, self.size), resample=Image.BICUBIC)
            image = np.array(image).astype(np.uint8)
            # center_crop 유지하되, 리사이즈 후라면 필요 없을 수도 있음
            # image = center_crop(self.size, self.size, image).astype(np.float32)
            image = (image / 127.5 - 1.0).astype(np.float32)
            image = einops.rearrange(image, 'h w c -> c h w')
            return image

        input_image = load_and_process(item["input_image"])
        input_text = item["input_text"]
        output_image = load_and_process(item["output_image"])
        output_text = item["output_text"]

        return input_image, input_text, output_image, output_text











import sys, os
sys.path.append(os.path.dirname(__file__))  # 현재 디렉토리를 import 경로에 추가
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from si_finetuning_2.utils import load_models

 # 클립/오토인코더/프로젝션 등 불러오는 함수
class DialogueSubDataset(Dataset):
    def __init__(self, json_path, image_root, autoencoder, clip_img_model, clip_text_model,
                 clip_preprocess, device, random_flip=False, p_uncond=None, linear_proj=None):
        print("[DEBUG] 데이터셋 init 시작")

        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        print("[DEBUG] json 파일 로딩 완료, 샘플 수:", len(self.samples))

        self.image_root = image_root
        self.p_uncond = p_uncond
        self.device = device


        print("[DEBUG] autoencoder 로딩 시작")
        from configs import finetune_uvit_config
        config = finetune_uvit_config.get_config()
        self.autoencoder = load_models(config)

        # dataset 설정에 넣어주기 (이미 잘 하셨음)
        config.dataset.autoencoder = self.autoencoder
        print("[DEBUG] autoencoder 로딩 완료")

        #self.autoencoder.to(device)
        self.autoencoder = autoencoder.eval()
        print("[DEBUG] autoencoder device 이동 및 eval 설정 완료")

        self.clip_img_model = clip_img_model.eval()
        self.clip_text_model = clip_text_model.eval()
        print("[DEBUG] CLIP 모델 eval 완료")

        self.linear_proj = linear_proj  # 학습 가능한 레이어
        #self.linear_proj = nn.Linear(768, 64).to(device)  # 768 → 64 projection
        print("[DEBUG] Linear projection 생성 완료")

        self.clip_preprocess = clip_preprocess

        transform = [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
        if random_flip:
            transform.insert(0, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        print("[DEBUG] get item 진입 ")
        item = self.samples[idx]
        print(f"[DEBUG] Loading sample {idx}: {item['input_image']} -> {item['output_image']}")

        # Load and preprocess input image
        input_img_pil = Image.open(os.path.join(self.image_root, item['input_image'])).convert("RGB")
        print(f"[DEBUG] Loaded input image {item['input_image']}")
        input_img_tensor = self.transform(input_img_pil).to(self.device)  # (3, 64, 64)
        input_img_clip = self.clip_preprocess(input_img_pil).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        print(f"[DEBUG] Transformed input image")

        # Load and preprocess output image (필요하면 latent도 같이 만듦)
        output_img_pil = Image.open(os.path.join(self.image_root, item['output_image'])).convert("RGB")
        output_img_tensor = self.transform(output_img_pil).to(self.device)
        print(f"[DEBUG] Loaded output image")

        input_text = item['input_text']
        output_text = item['output_text']

        if self.p_uncond is not None and random.random() < self.p_uncond:
            input_text = ""

        # 1) Autoencoder latent (z)
        with torch.no_grad():
            #z = self.autoencoder.encode(input_img_tensor.unsqueeze(0)).latent_dist.mean  # (1, 4, 64, 64)
            moments = self.autoencoder.encode_moments(input_img_tensor.unsqueeze(0))
            # moments: [mean, logvar]
            z = moments[0]  # mean

            print(f"🦖🦖🦖🦖🦖[DEBUG] Encoding type: {type(z)}🦖🦖🦖🦖🦖🦖")
            print(f"[DEBUG] Encoding shape: {z.shape if hasattr(z, 'shape') else 'N/A'}")

            #z = z.squeeze(0)

            print(f"🦖🦖🦖🦖🦖[DEBUG] Encoding type: {type(z)}🦖🦖🦖🦖🦖🦖")
            print(f"[DEBUG] Encoding shape: {z.shape if hasattr(z, 'shape') else 'N/A'}")
        print(f"[DEBUG] Encoded image with autoencoder")

        # 2) CLIP 이미지 임베딩
        with torch.no_grad():
            # clip_img = self.clip_img_model.encode_image(input_img_clip)  # (1, 512)
            # clip_img = clip_img.unsqueeze(1).squeeze(0)  # (1, 512)
            clip_img = self.clip_img_model.encode_image(input_img_clip).squeeze(0)  # (512,)
        print(f"[DEBUG] CLIP img embedding done")

        # 3) CLIP 텍스트 임베딩 + projection
        # tokens = self.clip_text_model.tokenize([input_text]).to(self.device)  # (1, 77)
        # #self.clip. FrozenCLIPEmbedder()
        # # 기존 잘못된 방식 (오류 발생)
        # tokens = self.clip_text_model.tokenize([input_text]).to(self.device)
        # with torch.no_grad():
        #     clip_text_feat = self.clip_text_model.encode_text(tokens)

        # ✅ 수정된 방식
        with torch.no_grad():
            clip_text_feat = self.clip_text_model.encode([input_text])  # (1, 768)

        #with torch.no_grad():
        #    clip_text_feat = self.clip_text_model.encode_text(tokens)  # (1, 77, 768)
        print(f"[DEBUG] input_text : {input_text}")
        print(f"[DEBUG] Projected text latent")

        # projection은 학습 가능하므로 no_grad 아님
        text_latent = self.linear_proj(clip_text_feat)  # (1, 77, 64)
        text_latent = text_latent.squeeze(0)

        return {
            "z": z,  # (4,64,64)
            "clip_img": clip_img,  # (1,512)
            "text": text_latent,  # (77,64)
            "output_img": output_img_tensor,
            "output_text": output_text,
        }

class DialogueDataset(DatasetFactory):
    def __init__(self, train_json, test_json, image_root, autoencoder, clip_img_model,
                 clip_text_model, clip_preprocess, device, linear_proj, p_uncond=None):
        super().__init__()
        self.train = DialogueSubDataset(
            train_json, image_root, autoencoder, clip_img_model,
            clip_text_model, clip_preprocess, device,
            random_flip=True, p_uncond=p_uncond, linear_proj=linear_proj
        )
        self.test = DialogueSubDataset(
            test_json, image_root, autoencoder, clip_img_model,
            clip_text_model, clip_preprocess, device,
            random_flip=False, p_uncond=None, linear_proj=linear_proj
        )
        print(f"[DialogueDataset] Train: {len(self.train)} samples, Test: {len(self.test)} samples")

    @property
    def data_shape(self):
        return 3, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet64_guided_diffusion.npz'









def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    elif name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    elif name == 'imagenet512_features':
        return ImageNet512Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'mscoco256_features':
        return MSCOCO256Features(**kwargs)
    elif name == 'avamerg':
        return DialogueDataset(**kwargs)
    else:
        raise NotImplementedError(name)
