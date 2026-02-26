import tkinter as tk
from tkinter import ttk, scrolledtext
import os
import threading
import logging
import json
import time
import gc
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import ToTensor, Normalize, Resize
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')  # Используем TkAgg для совместимости с tkinter
import matplotlib.pyplot as plt
import kagglehub

# --- Загрузка датасета ---
# Download latest version

# баги ккакашки
print("Проверка/скачивание датасета...")
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Path to dataset files:", path)

# баги ккакашки
# баги ккакашки
# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Пути и конфигурация ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используется устройство: {device}")

# Пути к данным, скачанным через kagglehub
BASE_PATH = path
# Исправленный путь к изображениям (учитывает двойную папку img_align_celeba)
IMG_FOLDER = os.path.join(BASE_PATH, "img_align_celeba", "img_align_celeba")
METADATA_PATH = os.path.join(BASE_PATH, "list_attr_celeba.csv")
CKPT_PATH = "checkpoints/styleclip_vae_checkpoint.pth"  # Путь для сохранения чекпоинтов

IMG_SIZE = 128
LATENT_DIM = 512
MAPPING_DIM = 1024
GRAD_CLIP_VALUE = 1.0
GAN_LOSS_WEIGHT = 0.5
VAE_LOSS_WEIGHT = 75.0
CLIP_LOSS_WEIGHT = 2.0
PROGRESSIVE_STEPS = {4: 400, 8: 800, 16: 1200, 32: 1600, 64: 2000, 128: 2400}

# --- Глобальные переменные для данных ---
df = None
image_paths = None
text_embeddings = None
train_loader = None
val_loader = None
clip_model = None
clip_processor = None


# --- Загрузка CLIP ---
def load_clip():
    global clip_model, clip_processor
    try:
        logger.info("Загрузка CLIP модели...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP модель успешно загружена.")
    except Exception as e:
        logger.error(f"Не удалось загрузить CLIP: {e}")
        clip_model = None
        clip_processor = None


# --- Утилиты ---
def safe_to_uint8(img_batch):
    if img_batch.min() >= 0.0:
        x = img_batch
    else:
        x = (img_batch * 0.5) + 0.5
    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
    return x


def prepare_img_for_plot(img_tensor):
    img = img_tensor.squeeze(0).detach().cpu()
    if img.min() < 0:
        img = (img * 0.5) + 0.5
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def exists_dataset():
    return os.path.exists(IMG_FOLDER) and os.path.exists(METADATA_PATH)


def load_state_dict_fuzzy(module: nn.Module, state_dict: dict, module_name: str = "module"):
    current = module.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in current and v.size() == current[k].size()}
    missing = [k for k in current.keys() if k not in state_dict]
    unexpected = [k for k in state_dict.keys() if
                  k not in current or (k in current and state_dict[k].size() != current[k].size())]
    current.update(filtered)
    module.load_state_dict(current, strict=False)
    logger.info(
        f"{module_name}: загружено {len(filtered)}/{len(state_dict)} слоёв | missing={len(missing)} | ignored={len(unexpected)}")


# --- Датасет ---
class CelebADataset(Dataset):
    def __init__(self, image_paths, text_embeddings):
        self.image_paths = image_paths
        self.text_embeddings = text_embeddings
        self.resize = Resize((224, 224))
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}. Пропускаем индекс {idx}.")
            # Бросаем IndexError, чтобы DataLoader пропустил этот элемент
            raise IndexError(f"Cannot load image at index {idx}") from e

        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)  # -> [-1,1]
        text_embedding = self.text_embeddings[idx]
        return image, text_embedding


def generate_text_prompt(row):
    attrs = []
    attrs.append("a male" if row['Male'] == 1 else "a female")
    if row['Smiling'] == 1: attrs.append("smiling")
    if row['Young'] == 1: attrs.append("young")
    if row['Eyeglasses'] == 1: attrs.append("wearing eyeglasses")
    if row['Blond_Hair'] == 1: attrs.append("with blond hair")
    if len(attrs) > 1:
        prompt = f"A photo of {', '.join(attrs)} person."
    else:
        prompt = f"A photo of {attrs[0]} person."
    return prompt


def build_dataloaders(batch_size: int):
    global df, image_paths, text_embeddings, train_loader, val_loader
    if not exists_dataset():
        logger.error("Датасет не найден. Проверьте IMG_FOLDER и METADATA_PATH.")
        return False

    # Читаем файл list_attr_celeba.csv
    logger.info(f"Чтение метаданных из {METADATA_PATH}...")
    df = pd.read_csv(METADATA_PATH) # pd.read_csv читает .csv файл
    # Исправление: убираем префикс img_align_celeba/ из image_id
    df['image_id'] = df['image_id'].apply(lambda x: os.path.basename(x))
    df.replace(-1, 0, inplace=True) # Преобразуем -1 в 0
    df['text_prompt'] = df.apply(generate_text_prompt, axis=1) # Генерируем промпты
    logger.info(f"DataFrame создан: {df.shape[0]} записей")

    image_paths = [os.path.join(IMG_FOLDER, fname) for fname in df['image_id'].tolist()]
    text_prompts = df['text_prompt'].tolist()

    embeddings_file = 'text_embeddings.npy'
    if os.path.exists(embeddings_file):
        logger.info("Найден сохранённый файл эмбеддингов. Загружаю...")
        text_embeddings_np = np.load(embeddings_file)
        text_embeddings_t = torch.tensor(text_embeddings_np, dtype=torch.float32)
    else:
        if clip_model is None:
            raise RuntimeError("CLIP недоступен, не могу предобработать текст.")
        logger.info("Предобработка текстов CLIP...")
        text_embeddings_list = []
        batch_size_clip = 1024
        for i in tqdm(range(0, len(text_prompts), batch_size_clip), desc="Обработка текстов"):
            chunk = text_prompts[i:i+batch_size_clip]
            with torch.no_grad():
                processed = clip_processor(text=chunk, return_tensors="pt", padding=True, truncation=True).to(device)
                # --- ИСПРАВЛЕНИЕ: Правильное извлечение и нормализация эмбеддингов ---
                text_outputs = clip_model.get_text_features(**processed)
                # Извлекаем эмбеддинг CLS токена (первый токен последовательности)
                raw_embs = text_outputs.last_hidden_state[:, 0, :]
                # Нормализуем
                embs = raw_embs / raw_embs.norm(dim=-1, keepdim=True)
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                text_embeddings_list.append(embs.cpu())
        text_embeddings_t = torch.cat(text_embeddings_list, dim=0)
        np.save(embeddings_file, text_embeddings_t.numpy())
        logger.info("Текстовые эмбеддинги сохранены.")

    x_train, x_val, y_train, y_val = train_test_split(
        image_paths, text_embeddings_t.numpy(), test_size=0.1, random_state=SEED
    )
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = CelebADataset(x_train, y_train_t)
    val_ds = CelebADataset(x_val, y_val_t)

    # --- ИСПРАВЛЕНИЕ: num_workers=0 для избежания проблем с файлами ---
    # --- ИСПРАВЛЕНИЕ: Выключаем pin_memory если нет GPU ---
    is_cuda_available = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=is_cuda_available)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=is_cuda_available)

    logger.info(f"Датасет загружен: Train: {len(train_ds)}, Val: {len(val_ds)}, Batch: {batch_size}")
    return True


# --- Модель ---
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            # Исправление: x.device -> x.device (добавлена 'e')
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        # Эти строки принадлежат __init__, а не forward
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02)
        self.style_dense = nn.Linear(MAPPING_DIM, in_channels)
        self.padding = kernel_size // 2

    def forward(self, x, w):
        b = x.shape[0]
        style = self.style_dense(w).unsqueeze(2).unsqueeze(3) # (B, in,1,1)
        modulated_kernel = self.kernel.unsqueeze(0) * style.unsqueeze(1) # (B, out, in, k, k)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x_reshaped = x.reshape(1, b * self.in_channels, x.shape[2], x.shape[3])
        kernel_reshaped = modulated_kernel.reshape(
            b * self.out_channels, self.in_channels, self.kernel.shape[2], self.kernel.shape[3])
        # ВАЖНО: явный padding числом, а не 'same' (совместимость PyTorch)
        out = F.conv2d(x_reshaped, kernel_reshaped, padding=self.padding, groups=b)
        out = out.reshape(b, self.out_channels, out.shape[2], out.shape[3])
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.to_rgb = ModulatedConv2d(in_channels, 3, 1)

    def forward(self, x, w):
        return self.to_rgb(x, w)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, upsample=upsample)
        self.noise1 = NoiseInjection()
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3)
        self.noise2 = NoiseInjection()
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        x = self.act1(self.noise1(self.conv1(x, w)))
        x = self.act2(self.noise2(self.conv2(x, w)))
        return x


class SynthesisNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.initial_constant = nn.Parameter(torch.randn(1, channels[0], 4, 4) * 0.02)
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(GeneratorBlock(channels[i], channels[i + 1], upsample=(i > 0)))
            self.to_rgbs.append(ToRGB(channels[i + 1]))

    def forward(self, w, resolution, alpha=1.0):
        x = self.initial_constant.expand(w.size(0), -1, -1, -1)
        rgb = None
        for i, res in enumerate([4, 8, 16, 32, 64, 128]):
            if res > resolution: break
            x = self.blocks[i](x, w[:, i])
            new_rgb = self.to_rgbs[i](x, w[:, i])
            if rgb is not None:
                if res == resolution and alpha < 1.0:
                    old_rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
                    rgb = (1 - alpha) * old_rgb + alpha * new_rgb
                else:
                    rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + new_rgb
            else:
                rgb = new_rgb
        return torch.tanh(rgb)


class CombinedMappingNetwork(nn.Module):
    def __init__(self, latent_dim, mapping_dim):
        super().__init__()
        self.num_layers = int(math.log2(IMG_SIZE)) * 2 - 2
        self.z_mapper = nn.Sequential(nn.Linear(latent_dim, mapping_dim), nn.LeakyReLU(0.2, inplace=True))

        self.text_mapper = nn.Sequential(nn.Linear(512, mapping_dim), nn.LeakyReLU(0.2, inplace=True))
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(mapping_dim, mapping_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.main_layers = nn.Sequential(*layers)

    def forward(self, z, text_embedding, alpha=0.5):
        z_mapped = self.z_mapper(z)
        text_mapped = self.text_mapper(text_embedding) # text_embedding теперь 768
        w_base = (1 - alpha) * z_mapped + alpha * text_mapped
        w = self.main_layers(w_base)
        w = w.unsqueeze(1).repeat(1, self.num_layers + 1, 1)
        w_mix = torch.randn_like(w)
        w_mix[:, :self.num_layers // 2] = w[:, :self.num_layers // 2]
        return w_mix


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 7, 1, 0), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.dense = nn.Linear(512, 1024)
        self.z_mean = nn.Linear(1024, latent_dim)
        self.z_log_var = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = self.layers(x)
        x = F.leaky_relu(self.dense(x), 0.2)
        return self.z_mean(x), self.z_log_var(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.down(x)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ch_rev = list(reversed(channels))
        self.from_rgbs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(ch_rev) - 1, 0, -1):
            self.from_rgbs.insert(0, nn.utils.spectral_norm(nn.Conv2d(3, ch_rev[i], 1)))
            self.blocks.insert(0, DiscriminatorBlock(ch_rev[i], ch_rev[i - 1]))
        self.from_rgbs.insert(0, nn.utils.spectral_norm(nn.Conv2d(3, ch_rev[0], 1)))
        self.final = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch_rev[0], ch_rev[0], 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(ch_rev[0] * 4 * 4, 1)
        )

    def forward(self, x, resolution, alpha=1.0):
        res_idx = int(math.log2(resolution)) - 2
        x_curr = self.from_rgbs[res_idx](x)
        if res_idx > 0 and alpha < 1.0:
            x_curr_processed = self.blocks[res_idx - 1](x_curr)
            downsampled_x = F.avg_pool2d(x, 2)
            x_prev = self.from_rgbs[res_idx - 1](downsampled_x)
            x = (1.0 - alpha) * x_prev + alpha * x_curr_processed
            for i in range(res_idx - 2, -1, -1):
                x = self.blocks[i](x)
        else:
            x = self.blocks[res_idx - 1](x_curr) if res_idx > 0 else x_curr
            for i in range(res_idx - 2, -1, -1):
                x = self.blocks[i](x)
        return self.final(x)


class StyleCLIPVAE(nn.Module):
    def __init__(self, latent_dim, mapping_dim, channels):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.mapping_network = CombinedMappingNetwork(latent_dim, mapping_dim)
        self.synthesis_network = SynthesisNetwork(channels)

    def generator(self, z, text_embedding, resolution, alpha=0.5):
        w = self.mapping_network(z, text_embedding, alpha)
        images = self.synthesis_network(w, resolution)
        return images

    def forward(self, x, text_embedding, resolution, alpha=0.5):
        z_mean, z_log_var = self.encoder(x)
        std = torch.exp(0.5 * z_log_var);
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        recon = self.generator(z, text_embedding, resolution, alpha)
        return recon, z_mean, z_log_var


# --- Функции потерь и градиентов ---
def generator_loss_gp(fake_pred):
    return -fake_pred.mean()


def discriminator_loss_gp(real_pred, fake_pred):
    return fake_pred.mean() - real_pred.mean()


def gradient_penalty(discriminator, real_images, fake_images, resolution, alpha, device):
    t = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    interpolates = (t * real_images + (1 - t) * fake_images).requires_grad_(True)
    disc_interpolates = discriminator(interpolates, resolution, alpha)
    grads = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# --- Обучение ---
# --- Обучение ---
def train_model(learning_rate, epochs, batch_size, progress_callback=None):
    gc.collect()
    torch.cuda.empty_cache()

    if not build_dataloaders(batch_size):
        logger.error("Не удалось загрузить датасет.")
        return

    CHANNELS = [512, 512, 256, 128, 64, 32, 16]
    model = StyleCLIPVAE(LATENT_DIM, MAPPING_DIM, CHANNELS).to(device)
    disc = Discriminator(CHANNELS).to(device)

    gen_optim = optim.Adam(
        list(model.encoder.parameters()) +
        list(model.mapping_network.parameters()) +
        list(model.synthesis_network.parameters()),
        lr=learning_rate, betas=(0.5, 0.999)
    )
    disc_optim = optim.Adam(disc.parameters(), lr=5e-6, betas=(0.5, 0.999))

    scheduler_g = CosineAnnealingLR(gen_optim, T_max=sum(PROGRESSIVE_STEPS.values()), eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(disc_optim, T_max=sum(PROGRESSIVE_STEPS.values()), eta_min=1e-7)
    recon_loss_fn = nn.L1Loss()
    # --- ИСПРАВЛЕНИЕ: Обработка GradScaler для CPU ---
    if device.type == 'cuda':
        scaler = GradScaler()
    else:
        class DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        scaler = DummyScaler()
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    start_epoch = 0
    current_resolution = 4
    total_steps = 0

    if os.path.exists(CKPT_PATH):
        logger.info("Найден чекпоинт. Загружаю...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        load_state_dict_fuzzy(model, ckpt['model_state_dict'], module_name="StyleCLIPVAE")
        if 'discriminator_state_dict' in ckpt:
            load_state_dict_fuzzy(disc, ckpt['discriminator_state_dict'], module_name="Discriminator")
        try:
            gen_optim.load_state_dict(ckpt['gen_optim_state_dict'])
            disc_optim.load_state_dict(ckpt['disc_optim_state_dict'])
        except Exception as e:
            logger.warning(f"Не удалось восстановить оптимайзеры: {e}. Продолжаем с текущими.")
        # --- ИСПРАВЛЕНИЕ: Загрузка scaler только если CUDA ---
        if 'scaler_state_dict' in ckpt and device.type == 'cuda':
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        start_epoch = ckpt.get('epoch', 0) + 1
        current_resolution = ckpt.get('current_resolution', 4)
        total_steps = ckpt.get('total_steps', 0)
        logger.info(f"Продолжаем с эпохи {start_epoch}, разрешения {current_resolution}x{current_resolution}")

    logger.info(f"Запуск обучения: эпох={epochs}, lr={learning_rate}, batch={batch_size}")

    for epoch in range(start_epoch, epochs):
        model.train()
        disc.train()
        total_recon = total_kl = total_gen = total_disc = total_clip = 0.0

        loop = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs} | Рез {current_resolution}")
        for i, (images, text_emb) in enumerate(loop):
            images = images.to(device, non_blocking=True)
            text_emb = text_emb.to(device, non_blocking=True)

            steps_needed = PROGRESSIVE_STEPS.get(current_resolution, float('inf'))
            alpha = min(1.0, (total_steps / steps_needed))

            if total_steps > sum(list(PROGRESSIVE_STEPS.values())[:int(math.log2(current_resolution) - 1)]):
                if current_resolution < IMG_SIZE:
                    current_resolution *= 2
                    logger.info(f"Переход на {current_resolution}x{current_resolution}")

            # --- Генератор ---
            gen_optim.zero_grad(set_to_none=True)
            with autocast() if device.type == 'cuda' else torch.no_grad(): # Используем autocast только с CUDA
                recon, z_mu, z_logvar = model(images, text_emb, current_resolution, alpha)
                with torch.no_grad():
                    real_resized = F.interpolate(images, size=(current_resolution, current_resolution), mode='bilinear',
                                                 align_corners=False)
                recon_loss = recon_loss_fn(recon, real_resized)
                kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
                vae_loss = VAE_LOSS_WEIGHT * (recon_loss + kl_loss)
                pred_fake = disc(recon, current_resolution, alpha)
                g_loss = GAN_LOSS_WEIGHT * generator_loss_gp(pred_fake)

                if clip_model is not None:
                    with torch.no_grad():
                        img_proc = (recon * 0.5) + 0.5
                        img_clip = F.interpolate(img_proc, size=(224, 224), mode='bilinear', align_corners=False)
                        # --- ИСПРАВЛЕНИЕ: Извлечение и нормализация эмбеддингов изображений ---
                        img_feat_raw = clip_model.get_image_features(pixel_values=img_clip)
                        # Извлекаем эмбеддинг CLS токена (первый токен последовательности)
                        img_feat_unnorm = img_feat_raw.last_hidden_state[:, 0, :]
                        # Нормализуем
                        img_feat = img_feat_unnorm / img_feat_unnorm.norm(dim=-1, keepdim=True)
                        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                        # text_emb уже нормализован (предполагается, что он нормализован в build_dataloaders)
                        text_norm = text_emb / text_emb.norm(dim=-1, keepdim=True)
                        clip_loss = 1.0 - torch.mean((img_feat * text_norm).sum(dim=-1))
                else:
                    clip_loss = torch.tensor(0.0, device=device)

                total_loss = vae_loss + g_loss + CLIP_LOSS_WEIGHT * clip_loss

            if torch.isfinite(total_loss):
                scaler.scale(total_loss).backward()
                scaler.unscale_(gen_optim)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                scaler.step(gen_optim)
                scaler.update()
            else:
                logger.warning("NaN/Inf в лоссе генератора, шаг пропущен.")

            # --- Дискриминатор ---
            disc_optim.zero_grad(set_to_none=True)
            with autocast() if device.type == 'cuda' else torch.no_grad(): # Используем autocast только с CUDA
                real_scaled = F.interpolate(images, size=(current_resolution, current_resolution), mode='bilinear',
                                            align_corners=False)
                pred_real = disc(real_scaled, current_resolution, alpha)
                z = torch.randn(real_scaled.size(0), LATENT_DIM, device=device)
                with torch.no_grad():
                    fake = model.generator(z, text_emb, current_resolution, alpha=0.5)
                pred_fake2 = disc(fake.detach(), current_resolution, alpha)
                d_loss = discriminator_loss_gp(pred_real, pred_fake2)
                gp = gradient_penalty(disc, real_scaled, fake.detach(), current_resolution, alpha, device)
                disc_loss = d_loss + 10.0 * gp

            if torch.isfinite(disc_loss):
                scaler.scale(disc_loss).backward()
                scaler.unscale_(disc_optim)
                nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP_VALUE)
                scaler.step(disc_optim)
                scaler.update()
            else:
                logger.warning("NaN/Inf в лоссе дискриминатора, шаг пропущен.")

            scheduler_g.step()
            scheduler_d.step()

            total_recon += float(recon_loss.item())
            total_kl += float(kl_loss.item())
            total_gen += float(g_loss.item())
            total_disc += float(disc_loss.item())
            total_clip += float(clip_loss.item()) if isinstance(clip_loss, torch.Tensor) else float(clip_loss)

            loop.set_postfix({
                "Recon": f"{recon_loss.item():.3f}",
                "KL": f"{kl_loss.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "D": f"{disc_loss.item():.3f}",
                "CLIP": f"{float(clip_loss):.3f}",
                "α": f"{alpha:.2f}",
            })
            total_steps += 1

        # Итоги эпохи
        n = len(train_loader)
        logger.info(f"Эпоха {epoch + 1}/{epochs} | Рез {current_resolution}x{current_resolution}")
        logger.info(
            f"Recon: {total_recon / n:.4f} | KL: {total_kl / n:.4f} | G: {total_gen / n:.4f} | D: {total_disc / n:.4f} | CLIP: {total_clip / n:.4f}")

        # Сохранение чекпоинта
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        # --- ИСПРАВЛЕНИЕ: Сохранение scaler только если CUDA ---
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'discriminator_state_dict': disc.state_dict(),
            'gen_optim_state_dict': gen_optim.state_dict(),
            'disc_optim_state_dict': disc_optim.state_dict(),
            'current_resolution': current_resolution,
            'total_steps': total_steps,
        }
        if device.type == 'cuda':
            save_dict['scaler_state_dict'] = scaler.state_dict()
        torch.save(save_dict, CKPT_PATH)
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        logger.info(f"Чекпоинт сохранён: {CKPT_PATH}")

        if progress_callback:
            progress_callback(epoch + 1, epochs)

    logger.info("Обучение завершено.")
# --- Метрики ---
@torch.no_grad()
def calculate_fid(model, val_loader, device, current_resolution, num_samples=1000):
    logger.info("Расчёт FID...")
    model.eval()
    fid = FrechetInceptionDistance(feature=64).to(device)

    seen = 0
    for images, _ in val_loader:
        if seen >= num_samples: break
        images = images.to(device)
        real_resized = F.interpolate(images, size=(current_resolution, current_resolution), mode='bilinear',
                                     align_corners=False)
        fid.update(safe_to_uint8(real_resized), real=True)
        seen += images.size(0)

    gen = 0
    while gen < num_samples:
        z = torch.randn(val_loader.batch_size, LATENT_DIM, device=device)
        text_embedding = torch.zeros(val_loader.batch_size, 512, device=device)
        fake = model.generator(z, text_embedding, current_resolution, alpha=0.5)
        fid.update(safe_to_uint8(fake), real=False)
        gen += val_loader.batch_size

    score = fid.compute().item()
    logger.info(f"FID: {score:.4f}")
    return score


@torch.no_grad()
def calculate_is(model, val_loader, device, current_resolution, num_samples=1000):
    logger.info("Расчёт Inception Score...")
    model.eval()
    inception = InceptionScore(normalize=True).to(device)
    generated_images = []
    gen_count = 0
    while gen_count < num_samples:
        z = torch.randn(val_loader.batch_size, LATENT_DIM, device=device)
        text_embedding = torch.zeros(val_loader.batch_size, 512, device=device)
        fake = model.generator(z, text_embedding, current_resolution, alpha=0.5)
        generated_images.append(fake.cpu())
        gen_count += val_loader.batch_size
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    norm_images = (generated_images * 0.5) + 0.5
    inception.update(norm_images.to(device))
    score_mean, score_std = inception.compute()
    logger.info(f"Inception Score: Mean={score_mean.item():.4f}, Std={score_std.item():.4f}")
    return score_mean.item(), score_std.item()


@torch.no_grad()
def calculate_clic(model, val_loader, clip_model, device, current_resolution, num_samples=1000):
    if clip_model is None:
        logger.warning("CLIP недоступен, пропускаю CLIC.")
        return float('nan')
    logger.info("Расчёт CLIC...")
    model.eval()
    total, count = 0.0, 0
    for images, text_embeddings in val_loader:
        if count >= num_samples: break
        images = images.to(device)
        text_embeddings = text_embeddings.to(device)
        z = torch.randn(images.size(0), LATENT_DIM, device=device)
        fake = model.generator(z, text_embeddings, current_resolution, alpha=0.5)
        img_proc = (fake * 0.5) + 0.5
        img_clip = F.interpolate(img_proc, size=(224, 224), mode='bilinear', align_corners=False)
        # --- ИСПРАВЛЕНИЕ: Извлечение и нормализация эмбеддингов изображений для CLIC ---
        img_feat_raw = clip_model.get_image_features(pixel_values=img_clip)
        # Извлекаем эмбеддинг CLS токена (первый токен последовательности)
        img_feat_unnorm = img_feat_raw.last_hidden_state[:, 0, :]
        # Нормализуем
        img_feat = img_feat_unnorm / img_feat_unnorm.norm(dim=-1, keepdim=True)
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        text_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        sim = (img_feat * text_norm).sum(dim=-1) # sim - это тензор размера (batch_size,)
        total += float(sim.mean().item()) * images.size(0) # Усредняем внутри батча, потом умножаем на его размер
        count += images.size(0)
    score = total / max(1, count) # Избегаем деления на 0
    logger.info(f"CLIC: {score:.4f}")
    return score


def run_metrics(batch_size, progress_callback=None):
    if not build_dataloaders(batch_size):
        logger.error("Не удалось загрузить валидационный датасет.")
        return

    CHANNELS = [512, 512, 256, 128, 64, 32, 16]
    model = StyleCLIPVAE(LATENT_DIM, MAPPING_DIM, CHANNELS).to(device)

    if not os.path.exists(CKPT_PATH):
        logger.error("Чекпоинт не найден. Сначала обучите модель.")
        return

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    current_resolution = ckpt.get('current_resolution', IMG_SIZE)

    fid = calculate_fid(model, val_loader, device, current_resolution, num_samples=min(1000, len(val_loader.dataset)))
    clic = calculate_clic(model, val_loader, clip_model, device, current_resolution,
                          num_samples=min(1000, len(val_loader.dataset)))
    is_mean, is_std = calculate_is(model, val_loader, device, current_resolution,
                                   num_samples=min(1000, len(val_loader.dataset)))

    metrics = {
        "FID": float(fid),
        "CLIC": float(clic),
        "IS_mean": float(is_mean),
        "IS_std": float(is_std),
        "resolution": int(current_resolution),
        "timestamp": int(time.time())
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Метрики сохранены в metrics.json")

    if progress_callback:
        progress_callback(100, 100)  # Условное завершение


# --- Визуализация ---
@torch.no_grad()
def plot_generated_images(model, prompts, current_resolution):
    model.eval()
    n = len(prompts)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for i, ax in enumerate(axes):
        text_inputs = clip_processor(text=[prompts[i]], return_tensors="pt", padding=True, truncation=True).to(device)
        text_embedding = clip_model.get_text_features(**text_inputs)
        z = torch.randn(1, LATENT_DIM, device=device)
        img = model.generator(z, text_embedding, current_resolution)
        ax.imshow(prepare_img_for_plot(img))
        ax.set_title(prompts[i], fontsize=9)
        ax.axis('off')
    plt.suptitle(f'Сгенерированные изображения @ {current_resolution}x{current_resolution}')
    plt.show()


@torch.no_grad()
def plot_latent_interpolation(model, prompt, current_resolution, steps=6):
    model.eval()
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    text_embedding = clip_model.get_text_features(**text_inputs)
    z1 = torch.randn(1, LATENT_DIM, device=device)
    z2 = torch.randn(1, LATENT_DIM, device=device)
    fig, axes = plt.subplots(1, steps, figsize=(4 * steps, 4))
    for i in range(steps):
        t = i / (steps - 1)
        z = (1 - t) * z1 + t * z2
        img = model.generator(z, text_embedding, current_resolution)
        axes[i].imshow(prepare_img_for_plot(img))
        axes[i].set_title(f"t={t:.2f}")
        axes[i].axis('off')
    plt.suptitle(f'Интерполяция латента "{prompt}"')
    plt.show()


@torch.no_grad()
def plot_attribute_manipulation(model, current_resolution):
    model.eval()
    z = torch.randn(1, LATENT_DIM, device=device)
    base_prompt = "A photo of a person."
    modified_prompt = "A photo of a smiling person."
    base_te = clip_model.get_text_features(
        **clip_processor(text=[base_prompt], return_tensors="pt", padding=True, truncation=True).to(device))
    modified_te = clip_model.get_text_features(
        **clip_processor(text=[modified_prompt], return_tensors="pt", padding=True, truncation=True).to(device))
    base_img = model.generator(z, base_te, current_resolution, alpha=0.5)
    modified_img = model.generator(z, modified_te, current_resolution, alpha=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(prepare_img_for_plot(base_img));
    axes[0].set_title(f"База: {base_prompt}", fontsize=9);
    axes[0].axis('off')
    axes[1].imshow(prepare_img_for_plot(modified_img));
    axes[1].set_title(f"Изменено: {modified_prompt}", fontsize=9);
    axes[1].axis('off')
    plt.suptitle('Манипуляция атрибутом "улыбка"')
    plt.show()


def run_generation(prompts_str, progress_callback=None):
    if clip_model is None:
        logger.warning("CLIP недоступен — визуализации отключены.")
        return

    if not os.path.exists(CKPT_PATH):
        logger.error("Чекпоинт не найден. Сначала обучите модель.")
        return

    CHANNELS = [512, 512, 256, 128, 64, 32, 16]
    model = StyleCLIPVAE(LATENT_DIM, MAPPING_DIM, CHANNELS).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    current_resolution = ckpt.get('current_resolution', IMG_SIZE)

    prompts = [p.strip() for p in prompts_str.split("\n") if p.strip()]
    if len(prompts) == 0:
        prompts = ["A photo of a person."]

    plot_generated_images(model, prompts, current_resolution)
    plot_latent_interpolation(model, prompts[0], current_resolution, steps=6)
    plot_attribute_manipulation(model, current_resolution)

    if progress_callback:
        progress_callback(100, 100)  # Условное завершение


# --- Класс приложения Tkinter ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("StyleCLIP-VAE Trainer & Generator")

        # Переменные для параметров
        self.learning_rate_var = tk.DoubleVar(value=5e-5)
        self.epochs_var = tk.IntVar(value=1)
        self.batch_size_var = tk.IntVar(value=128)
        self.prompts_var = tk.StringVar(value="A photo of a young female with blond hair.\nA photo of a smiling male.")

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Панель параметров ---
        params_frame = ttk.LabelFrame(main_frame, text="Параметры", padding="5")
        params_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W)
        lr_entry = ttk.Entry(params_frame, textvariable=self.learning_rate_var)
        lr_entry.grid(row=0, column=1, padx=(5, 10))

        ttk.Label(params_frame, text="Эпохи:").grid(row=0, column=2, sticky=tk.W)
        epochs_spin = ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.epochs_var)
        epochs_spin.grid(row=0, column=3, padx=(5, 10))

        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=4, sticky=tk.W)
        bs_spin = ttk.Spinbox(params_frame, from_=16, to=256, increment=16, textvariable=self.batch_size_var)
        bs_spin.grid(row=0, column=5, padx=(5, 10))

        # --- Поле для промптов ---
        ttk.Label(main_frame, text="Промпты (по одному в строке):").grid(row=1, column=0, sticky=(tk.W, tk.E),
                                                                         pady=(0, 5))
        self.prompts_text = scrolledtext.ScrolledText(main_frame, height=4, width=70)
        self.prompts_text.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        self.prompts_text.insert(tk.END, self.prompts_var.get())

        # --- Кнопки ---
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))

        self.btn_train = ttk.Button(buttons_frame, text="▶️ Обучить модель", command=self.start_train_thread)
        self.btn_train.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_metrics = ttk.Button(buttons_frame, text="📈 Посчитать метрики", command=self.start_metrics_thread)
        self.btn_metrics.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_generate = ttk.Button(buttons_frame, text="🖼️ Сгенерировать", command=self.start_generate_thread)
        self.btn_generate.pack(side=tk.LEFT, padx=(0, 5))

        # --- Прогресс бар ---
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=(10, 5))

        # --- Логи ---
        ttk.Label(main_frame, text="Логи:").grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.log_text = scrolledtext.ScrolledText(main_frame, height=10, width=70, state='disabled')
        self.log_text.grid(row=6, column=0, columnspan=2)

    def log_message(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def update_progress(self, current, total):
        percentage = (current / total) * 100
        self.progress['value'] = percentage
        self.root.update_idletasks()

    def start_train_thread(self):
        thread = threading.Thread(target=self.run_train)
        thread.daemon = True
        thread.start()

    def run_train(self):
        lr = self.learning_rate_var.get()
        epochs = self.epochs_var.get()
        bs = self.batch_size_var.get()
        self.log_message(f"Начинаю обучение с lr={lr}, epochs={epochs}, batch_size={bs}...")
        train_model(lr, epochs, bs, progress_callback=self.update_progress)
        self.log_message("Обучение завершено.")

    def start_metrics_thread(self):
        thread = threading.Thread(target=self.run_metrics)
        thread.daemon = True
        thread.start()

    def run_metrics(self):
        bs = self.batch_size_var.get()
        self.log_message(f"Начинаю расчёт метрик с batch_size={bs}...")
        run_metrics(bs, progress_callback=self.update_progress)
        self.log_message("Расчёт метрик завершён.")

    def start_generate_thread(self):
        thread = threading.Thread(target=self.run_generate)
        thread.daemon = True
        thread.start()

    def run_generate(self):
        prompts_str = self.prompts_text.get("1.0", tk.END).strip()
        self.log_message(f"Начинаю генерацию изображений...")
        run_generation(prompts_str, progress_callback=self.update_progress)
        self.log_message("Генерация завершена.")


if __name__ == "__main__":
    load_clip()  # Загружаем CLIP при запуске
    root = tk.Tk()
    app = App(root)
    root.mainloop()