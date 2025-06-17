import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchvision import transforms
from PIL import Image
from sample_multi_v1 import evaluate
from libs.caption_decoder import CaptionDecoder
import importlib.util
from absl import flags
from absl import app
from ml_collections import config_flags
import os
import time

# 사용자 정의 데이터셋
class AvaMERGDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_root):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_image = self.transform(Image.open(os.path.join(self.image_root, item['input_image'])).convert('RGB'))
        output_image = self.transform(Image.open(os.path.join(self.image_root, item['output_image'])).convert('RGB'))
        return {
            'input_image': input_image,
            'input_text': item['input_text'],
            'output_image': output_image,
            'output_text': item['output_text']
        }

# BLEU 계산 함수
bleu_smoother = SmoothingFunction().method4
def compute_bleu(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=bleu_smoother)

# FID 계산 함수 구현 (clean-fid 또는 pytorch-fid 필요)
def calculate_fid_given_paths(paths, batch_size=32, device='cuda', dims=2048):
    try:
        from cleanfid import fid
        return fid.compute_fid(paths[0], paths[1], batch_size=batch_size, device=device, mode="clean")
    except ImportError:
        print("[!] clean-fid가 설치되어 있지 않습니다. 다음 명령으로 설치하세요: pip install clean-fid")
        return float('nan')

# 메인 평가 루프
def evaluate_model(config, json_path, image_root, output_dir, batch_size=4):
    dataset = AvaMERGDataset(json_path, image_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pred_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gt_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'text_pairs'), exist_ok=True)

    all_bleu = []
    pred_image_paths, gt_image_paths = [], []

    for i, batch in enumerate(tqdm(dataloader)):
        for j in range(batch_size):
            idx = i * batch_size + j
            config.mode = 'joint'
            config.n_samples = 1
            config.prompt = batch['input_text'][j]
            config.seed = int(time.time()) + idx  # ← 랜덤성 부여 (중복 방지)

            evaluate(config)

            preds_path = os.path.join(config.output_path, 'joint')
            pred_img = Image.open(os.path.join(preds_path, '0.png')).convert('RGB')
            pred_img_path = os.path.join(output_dir, 'pred_images', f'{idx}.png')
            pred_img.save(pred_img_path)
            pred_image_paths.append(pred_img_path)

            gt_img = batch['output_image'][j]
            gt_img_path = os.path.join(output_dir, 'gt_images', f'{idx}.png')
            save_image(gt_img, gt_img_path)
            gt_image_paths.append(gt_img_path)

            with open(os.path.join(preds_path, 'prompts.txt'), 'r') as f:
                generated_text = f.readline().strip()

            input_text = batch['input_text'][j]
            output_text = batch['output_text'][j]
            text_pair_path = os.path.join(output_dir, 'text_pairs', f'{idx}.txt')
            with open(text_pair_path, 'w') as tf:
                tf.write(f"[INPUT] {input_text}\n")
                tf.write(f"[TARGET] {output_text}\n")
                tf.write(f"[GENERATED] {generated_text}\n")

            bleu = compute_bleu(output_text, generated_text)
            all_bleu.append(bleu)
            print(f"[INPUT] {input_text}\n")
            print(f"[TARGET] {output_text}\n")
            print(f"[GENERATED] {generated_text}\n")
            print(f"[Sample {idx}] BLEU: {bleu:.6f}")

        # 중간 FID 계산
        if len(pred_image_paths) >= batch_size:
            fid = calculate_fid_given_paths([os.path.join(output_dir, 'gt_images'),
                                             os.path.join(output_dir, 'pred_images')],
                                            batch_size=32, device='cuda', dims=2048)
            print(f"[Batch {i}] Current FID: {fid:.6f}")

    # 최종 FID 계산 및 저장
    fid = calculate_fid_given_paths([os.path.join(output_dir, 'gt_images'),
                                     os.path.join(output_dir, 'pred_images')],
                                     batch_size=32, device='cuda', dims=2048)

    results = {
        'BLEU': float(np.mean(all_bleu)),
        'FID': float(fid)
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("\n📊 Evaluation Complete")
    print(f"Average BLEU: {results['BLEU']:.6f}")
    print(f"FID: {results['FID']:.6f}")

def load_config_from_py(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

FLAGS = flags.FLAGS

# 필요한 flags만 정의 (중복 체크)
def define_flag_if_not_exists(flag_type, name, default, help_text):
    try:
        if flag_type == 'string':
            flags.DEFINE_string(name, default, help_text)
        elif flag_type == 'integer':
            flags.DEFINE_integer(name, default, help_text)
    except flags._exceptions.DuplicateFlagError:
        pass  # 이미 정의되어 있으면 무시


# 필요한 flags 정의
define_flag_if_not_exists('string', "nnet_path",
                          "/home/ivpl-d29/sichoi/Emo/unidiffuser/si_finetuning_2/leg_checkpoint_step_4000.pth",
                          "The nnet to evaluate.")
define_flag_if_not_exists('string', "output_path", "out", "dir to write results to")
define_flag_if_not_exists('string', "prompt", "an elephant under the sea",
                          "the prompt for text-to-image generation and text variation")
define_flag_if_not_exists('string', "img", "assets/space.jpg",
                          "the image path for image-to-text generation and image variation")
define_flag_if_not_exists('integer', "n_samples", 1, "the number of samples to generate")
define_flag_if_not_exists('integer', "nrow", 4, "number of images displayed in each row of the grid")
define_flag_if_not_exists('string', "mode", None, "type of generation, one of t2i / i2t / joint / i / t / i2t2i/ t2i2t")


def main(argv):
    # config 파일에서 기본 설정 불러오기
    config_path = 'configs/sample_unidiffuser_v1.py'
    config = load_config_from_py(config_path)

    # FLAGS로부터 추가 설정 업데이트
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.prompt = FLAGS.prompt
    config.nrow = min(FLAGS.nrow, FLAGS.n_samples)
    config.img = FLAGS.img
    config.n_samples = FLAGS.n_samples
    config.mode = FLAGS.mode

    # 추가 설정
    json_path = "/home/ivpl-d29/dataset/AvaMERG/test_finetune.json"
    image_root = "/home/ivpl-d29/dataset/AvaMERG/image_v5_0/"
    output_dir = "eval_output"
    batch_size = 4

    evaluate_model(config, json_path, image_root, output_dir, batch_size)


if __name__ == '__main__':
    app.run(main)