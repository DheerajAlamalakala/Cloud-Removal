"""
Dataset loading for cloud-removal CycleGAN training.

Expected folder layout (unpaired-style, standard CycleGAN convention):

    data_root/
        trainA/   <- cloudy images   (domain A)
        trainB/   <- cloud-free images (domain B)
        testA/    <- cloudy images for evaluation
        testB/    <- cloud-free images for evaluation

This works directly for:
  * RICE dataset (Kaggle) -> put "cloud" images in trainA/, "label"
    (cloud-free) images in trainB/.
  * SEN12MS-CR -> the release ships tiles as GeoTIFFs organized by scene
    (s1/, s2_cloudy/, s2_cloudfree/). Use `preprocess_sen12mscr.py` (see
    below) to export the s2_cloudy/ and s2_cloudfree/ folders as PNG/TIFF
    patches into trainA/ and trainB/ respectively, or point --channels to
    13 and load the GeoTIFFs directly by extending CloudDataset._load_image.

Because CycleGAN trains on unpaired data, images in trainA and trainB do
NOT need to correspond 1:1 or be listed in the same order -- the loader
randomly pairs a cloudy image with a clear image on every access. If your
data happens to be paired (e.g. RICE, or SEN12MS-CR cloudy/cloudfree
pairs), that's fine too -- CycleGAN just won't exploit the pairing, which
is intentional (this lets the same code handle unpaired real-world
collections too).
"""

import os
import random
import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(folder):
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


class CloudDataset(Dataset):
    """
    Unpaired image dataset for CycleGAN.

    domain_A_dir: folder of cloudy images
    domain_B_dir: folder of cloud-free images
    """

    def __init__(self, domain_A_dir, domain_B_dir, image_size=256, channels=3, mode="train", paired=True, mask_dir=None,
                 strong_aug=False, strict_pairing=True):
        self.files_A = list_images(domain_A_dir)
        self.files_B = list_images(domain_B_dir)
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise RuntimeError(
                f"No images found. Checked:\n  A: {domain_A_dir} ({len(self.files_A)} files)\n"
                f"  B: {domain_B_dir} ({len(self.files_B)} files)\n"
                "Make sure the dataset has been downloaded/extracted into these folders."
            )
        self.files_M = list_images(mask_dir) if mask_dir else []
        if paired or mode != "train":
            self._check_pairing(strict_pairing)
        if self.files_M and len(self.files_M) != len(self.files_A):
            raise RuntimeError(
                f"Mask count ({len(self.files_M)}) != cloudy count ({len(self.files_A)}). "
                "trainA, trainB and trainM must have matching filenames.")
        self.channels = channels
        self.mode = mode
        self.paired = paired
        self.image_size = image_size
        self.strong_aug = strong_aug  # also random vertical flip + 90-degree rotations (satellite imagery has no "up")

        aug = []
        if mode == "train":
            aug += [
                T.Resize(int(image_size * 1.12), Image.BICUBIC),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
            ]
        else:
            aug += [T.Resize((image_size, image_size), Image.BICUBIC)]

        self.norm = T.Compose([T.ToTensor(), T.Normalize([0.5] * channels, [0.5] * channels)])
        aug += [T.ToTensor()]
        # Normalize to [-1, 1] per channel (matches generator's final Tanh)
        aug += [T.Normalize([0.5] * channels, [0.5] * channels)]
        self.transform = T.Compose(aug)

    def _check_pairing(self, strict):
        """Paired mode pairs files by sorted position, so make sure that is really the same scene."""
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        problems = []
        if len(self.files_A) != len(self.files_B):
            problems.append(f"cloudy count ({len(self.files_A)}) != clear count ({len(self.files_B)})")
        else:
            bad = [(stem(a), stem(b)) for a, b in zip(self.files_A, self.files_B) if stem(a) != stem(b)]
            if bad:
                problems.append(f"{len(bad)} cloudy/clear filename mismatches, e.g. {bad[:3]}")
        if self.files_M and len(self.files_M) == len(self.files_A):
            badm = [(stem(a), stem(m)) for a, m in zip(self.files_A, self.files_M) if stem(a) != stem(m)]
            if badm:
                problems.append(f"{len(badm)} cloudy/mask filename mismatches, e.g. {badm[:3]}")
        if problems:
            msg = "Paired-data check failed: " + "; ".join(problems)
            if strict:
                raise RuntimeError(msg + "\nFix the file names, or pass --no_strict_pairing to only warn.")
            print("WARNING: " + msg)

    def __len__(self):
        # One epoch = one pass over the larger of the two domains
        return max(len(self.files_A), len(self.files_B))

    def _load_image(self, path):
        if self.channels == 3:
            img = Image.open(path).convert("RGB")
            return img
        else:
            # Multispectral path (e.g. pre-exported numpy/tiff stacks with >3 bands).
            # Expects a .npy file with shape (H, W, C) or a multi-band TIFF readable by PIL/rasterio.
            if path.endswith(".npy"):
                arr = np.load(path).astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                img = Image.fromarray((arr * 255).astype(np.uint8))
                return img
            img = Image.open(path)
            return img

    def _paired_transform(self, img_A, img_B, img_M=None):
        """Same resize / crop / flip on all images so pixels stay aligned."""
        IM = T.InterpolationMode
        if self.mode == "train":
            big = int(self.image_size * 1.12)
            img_A = TF.resize(img_A, big, IM.BICUBIC)
            img_B = TF.resize(img_B, big, IM.BICUBIC)
            if img_M is not None:
                img_M = TF.resize(img_M, big, IM.NEAREST)
            i, j, h, w = T.RandomCrop.get_params(img_A, (self.image_size, self.image_size))
            img_A, img_B = TF.crop(img_A, i, j, h, w), TF.crop(img_B, i, j, h, w)
            if img_M is not None:
                img_M = TF.crop(img_M, i, j, h, w)
            if random.random() < 0.5:
                img_A, img_B = TF.hflip(img_A), TF.hflip(img_B)
                if img_M is not None:
                    img_M = TF.hflip(img_M)
            if self.strong_aug:
                if random.random() < 0.5:
                    img_A, img_B = TF.vflip(img_A), TF.vflip(img_B)
                    if img_M is not None:
                        img_M = TF.vflip(img_M)
                k = random.randint(0, 3)
                if k:
                    img_A, img_B = TF.rotate(img_A, 90 * k), TF.rotate(img_B, 90 * k)
                    if img_M is not None:
                        img_M = TF.rotate(img_M, 90 * k)
        else:
            size = (self.image_size, self.image_size)
            img_A = TF.resize(img_A, size, IM.BICUBIC)
            img_B = TF.resize(img_B, size, IM.BICUBIC)
            if img_M is not None:
                img_M = TF.resize(img_M, size, IM.NEAREST)
        item_M = None
        if img_M is not None:
            # any non-black pixel (white / blue / yellow ...) = affected area -> 1
            m = np.array(img_M.convert("RGB")).max(axis=2) > 0
            item_M = torch.from_numpy(m.astype(np.float32))[None]
        return self.norm(img_A), self.norm(img_B), item_M

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        if self.paired or self.mode != "train":
            path_B = self.files_B[index % len(self.files_B)]
        else:
            path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]

        img_A = self._load_image(path_A)
        img_B = self._load_image(path_B)

        item = {"path_A": path_A, "path_B": path_B}
        if self.paired:
            img_M = Image.open(self.files_M[index % len(self.files_M)]) if self.files_M else None
            item_A, item_B, item_M = self._paired_transform(img_A, img_B, img_M)
            if item_M is not None:
                item["M"] = item_M
        else:
            item_A, item_B = self.transform(img_A), self.transform(img_B)

        item["A"], item["B"] = item_A, item_B
        return item


class ImagePool:
    """
    Buffer that stores previously generated images and, with 50% probability,
    feeds a historical image instead of the newest one to the discriminator.
    This is the trick from the original CycleGAN paper that stabilizes
    training and reduces oscillation.
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.detach(), 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)
