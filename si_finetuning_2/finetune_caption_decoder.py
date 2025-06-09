import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import default_data_collator
from transformers import EarlyStoppingCallback

# 데이터 collator와 조기 종료 콜백 설정
data_collator = default_data_collator
es = EarlyStoppingCallback(early_stopping_patience=5)  # 5 에포크 동안 개선이 없으면 조기 종료
import json
import argparse
from typing import Union, Optional
from collections import OrderedDict


# CLIP 특징을 입력으로 받아 GPT-2로 캡션을 생성하는 메인 모델 클래스
class ClipCaptionModel(nn.Module):
    """
    CLIP feature를 입력으로 받아 GPT-2를 이용해 캡션(텍스트)을 생성하는 모델.
    입력된 CLIP feature를 prefix로 사용하여 GPT-2 텍스트 생성을 조건화함.

    Architecture:
    1. CLIP features -> Linear layers (encode/decode) -> GPT-2 embedding space
    2. Concatenate with text embeddings -> GPT-2 forward pass
    3. Generate caption text using beam search or sampling
    """

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Loss 계산 시 prefix에 해당하는 부분을 무시하기 위한 더미 토큰 생성

        Args:
            batch_size: 배치 크기
            device: 텐서가 위치할 디바이스

        Returns:
            더미 토큰 텐서 (모두 0으로 채워짐)
        """
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        모델의 순전파 함수
        CLIP feature(prefix)와 텍스트 토큰을 입력받아 캡션 생성

        Args:
            tokens: GPT-2 입력 토큰 시퀀스 [batch_size, max_seq_len]
            prefix: CLIP 특징 벡터 [batch_size, prefix_length, 768]
            mask: 어텐션 마스크 (옵션)
            labels: 손실 계산용 라벨 (훈련 시 사용)

        Returns:
            GPT-2 모델의 출력 (logits, loss 등 포함)
        """
        # 1. 토큰을 GPT-2 임베딩 공간으로 변환
        embedding_text = self.gpt.transformer.wte(tokens)

        # 2. CLIP feature를 GPT-2 임베딩 공간으로 매핑
        # encode: 768 -> hidden_dim (선택적), decode: hidden_dim -> 768
        hidden = self.encode_prefix(prefix)
        prefix = self.decode_prefix(hidden)

        # 3. Prefix 임베딩과 텍스트 임베딩을 시퀀스 차원에서 연결
        # [batch, prefix_len + seq_len, embed_dim]
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)

        # 4. 훈련 시 라벨 처리 (prefix 부분은 손실 계산에서 제외)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        # 5. GPT-2 모델에 임베딩 입력하여 결과 생성
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)

        return out

    def encode_decode_prefix(self, prefix):
        """
        Prefix를 encode 후 다시 decode하여 원래 공간으로 복원
        주로 검증이나 디버깅 목적으로 사용
        """
        return self.decode_prefix(self.encode_prefix(prefix))

    def __init__(self, prefix_length: int, hidden_dim=None):
        """
        모델 초기화

        Args:
            prefix_length: CLIP feature의 시퀀스 길이 (보통 77)
            hidden_dim: 중간 은닉층 차원 (None이면 Identity 매핑)
        """
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length

        # 1. GPT-2 토크나이저 설정 및 특수 토큰 추가
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        base_tokenizer.add_special_tokens(special_tokens_dict)

        # 2. GPT-2 모델 초기화 (언어 모델링 헤드 포함)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2', eos_token_id=base_tokenizer.eos_token_id)
        self.gpt.resize_token_embeddings(len(base_tokenizer))  # 새 토큰에 맞게 임베딩 크기 조정

        # 3. Prefix 인코딩/디코딩 레이어 설정
        self.hidden_dim = hidden_dim
        # hidden_dim이 있으면 Linear 변환, 없으면 Identity (그대로 통과)
        self.encode_prefix = nn.Linear(768, hidden_dim) if hidden_dim is not None else nn.Identity()
        self.decode_prefix = nn.Linear(hidden_dim, 768) if hidden_dim is not None else nn.Identity()


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    """
    설정 파일과 체크포인트에서 모델을 로드하는 함수

    Args:
        config_path: JSON 설정 파일 경로
        epoch_or_latest: 로드할 에포크 번호 또는 '_latest'

    Returns:
        (model, parser): 로드된 모델과 설정 파서
    """
    # 설정 파일 로드
    with open(config_path) as f:
        config = json.load(f)

    # argparse를 사용해 설정을 파싱
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()

    # 모델 파일 경로 생성
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")

    # 모델 초기화 및 가중치 로드
    model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")

    return model, parser


def generate_beam(
        model,
        tokenizer,
        beam_size: int = 5,  # beam search에서 유지할 후보 시퀀스 수
        prompt=None,  # 텍스트 프롬프트 (보통 사용하지 않음)
        embed=None,  # 입력 임베딩 (CLIP에서 변환된 결과)
        entry_length=67,  # 생성할 최대 토큰 수
        temperature=1.0,  # 소프트맥스 온도 (낮을수록 더 결정적)
        stop_token: str = '<|EOS|>',  # 생성 종료 토큰
):
    """
    Beam Search를 사용해 텍스트 생성
    여러 후보 시퀀스를 동시에 탐색하여 가장 좋은 결과를 선택

    Args:
        model: 텍스트 생성 모델
        tokenizer: 토크나이저
        beam_size: 동시에 탐색할 후보 수
        embed: 입력 임베딩 (CLIP feature에서 변환)
        entry_length: 최대 생성 길이
        temperature: 생성 시 무작위성 조절
        stop_token: 생성 중단 토큰

    Returns:
        생성된 텍스트 리스트 (점수 순으로 정렬)
    """
    model.eval()  # 평가 모드로 설정

    # 초기화
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)  # 각 시퀀스 길이 추적
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)  # 종료 여부 추적

    with torch.no_grad():
        # 초기 임베딩 설정
        if embed is not None:
            generated = embed  # CLIP feature 기반 임베딩 사용
        else:
            # 텍스트 프롬프트를 임베딩으로 변환
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

        # 단계별 토큰 생성
        for i in range(entry_length):
            # 1. 모델 forward pass로 다음 토큰 확률 계산
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()  # log probability로 변환

            if scores is None:
                # 첫 번째 단계: 최상위 k개 토큰으로 beam 초기화
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)

                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                # 후속 단계: 기존 beam들을 확장
                logits[is_stopped] = -float(np.inf)  # 종료된 beam은 확장하지 않음
                logits[is_stopped, 0] = 0

                # 누적 점수 계산 및 길이로 정규화
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]

                # 최상위 k개 후보 선택
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]  # 어떤 beam에서 나왔는지
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)

                # 선택된 beam들로 상태 업데이트
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            # 새 토큰을 임베딩하고 시퀀스에 추가
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)

            # 종료 조건 확인
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    # 결과 정리 및 반환
    scores = scores / seq_lengths  # 길이로 정규화
    output_list = tokens.cpu().numpy()

    # 토큰을 텍스트로 디코딩
    output_texts = [
        tokenizer.decode(output[: int(length)], skip_special_tokens=True)
        for output, length in zip(output_list, seq_lengths)
    ]

    # 점수 기준 정렬
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    model.train()  # 다시 훈련 모드로 복원
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # 최대 생성 단어 수
        top_p=0.8,  # nucleus sampling 확률 임계값
        temperature=1.0,  # 생성 시 무작위성
        stop_token: str = '<|EOS|>',
):
    """
    Top-p (nucleus) sampling을 사용한 텍스트 생성
    확률 분포의 상위 p% 내에서만 샘플링하여 품질과 다양성의 균형을 맞춤

    Args:
        model: 생성 모델
        tokenizer: 토크나이저
        embed: 입력 임베딩
        entry_count: 생성할 시퀀스 개수
        entry_length: 최대 생성 길이
        top_p: nucleus sampling 임계값
        temperature: 생성 무작위성
        stop_token: 종료 토큰

    Returns:
        생성된 텍스트
    """
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")  # 제외할 토큰들의 확률값
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            # 초기 임베딩 설정
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

            # 토큰별 순차 생성
            for i in range(entry_length):
                # 다음 토큰 확률 계산
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                # Top-p sampling 적용
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )

                # 누적 확률이 top_p를 초과하는 토큰들 제거
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                # 다음 토큰 선택 (argmax 사용)
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                # 토큰 시퀀스 업데이트
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

                # 종료 토큰 확인
                if stop_token_index == next_token.item():
                    break

            # 결과 텍스트로 변환
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


class CaptionDecoder(object):
    """
    CLIP 특징을 입력받아 자연어 캡션을 생성하는 디코더 클래스
    사전 훈련된 모델을 로드하고 추론 및 파인튜닝 기능을 제공
    """

    def __init__(self, device, pretrained_path, hidden_dim=-1):
        """
        캡션 디코더 초기화

        Args:
            device: 모델이 실행될 디바이스 (cuda/cpu)
            pretrained_path: 사전 훈련된 모델 가중치 파일 경로
            hidden_dim: 은닉 차원 (-1이면 None으로 설정)
        """
        # hidden_dim 설정
        if hidden_dim < 0:
            hidden_dim = None

        # 1. 토크나이저 초기화
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 모델 초기화
        feature_length = 77  # CLIP feature 길이
        self.caption_model = ClipCaptionModel(feature_length, hidden_dim=hidden_dim)
        self.caption_model = ClipCaptionModel(feature_length, hidden_dim=None)  # 예시에서는 None 사용

        # 3. 사전 훈련된 가중치 로드
        print("Loading pretrained model...")
        ckpt = torch.load(pretrained_path, map_location='cpu')

        # DDP 모델에서 저장된 경우 'module.' prefix 제거
        filtered_state_dict = OrderedDict()
        for k, v in ckpt.items():
            key = k[7:] if k.startswith('module.') else k

            # 필요한 키만 필터링 (prefix, gpt 관련)
            if key.startswith('prefix') or key.startswith('gpt'):
                filtered_state_dict[key] = v

        # 모델에 가중치 로드
        missing_keys, unexpected_keys = self.caption_model.load_state_dict(filtered_state_dict, strict=False)

        # 로딩 결과 확인
        if len(missing_keys) > 0:
            print("Missing keys:", missing_keys)
        if len(unexpected_keys) > 0:
            print("Unexpected keys:", unexpected_keys)

        assert len(missing_keys) == 0  # 필수 가중치가 빠지면 안됨

        # unexpected_keys 검증 (clip, prefix, gpt, transformer 관련만 허용)
        allowed_prefixes = ('clip', 'prefix', 'gpt', 'transformer')
        assert all([name.startswith(allowed_prefixes) for name in unexpected_keys])

        # 4. 모델 설정 완료
        self.caption_model.eval()
        self.caption_model.to(device)
        self.caption_model.requires_grad_(False)  # 추론 모드에서는 기울기 계산 비활성화
        self.device = device

    def encode_prefix(self, features):
        """
        CLIP feature를 prefix embedding으로 인코딩

        Args:
            features: CLIP feature 텐서

        Returns:
            인코딩된 prefix embedding
        """
        return self.caption_model.encode_prefix(features)

    def generate_captions(self, features):
        """
        CLIP feature의 저차원 표현으로부터 캡션 생성

        Args:
            features: 인코딩된 CLIP feature [batch_size, feature_dim, embed_dim]

        Returns:
            생성된 캡션 텍스트 리스트
        """
        # 생성 설정
        use_beam_search = True

        # 배치를 개별 feature로 분할 (한 번에 하나씩 처리)
        features = torch.split(features, 1, dim=0)
        generated_captions = []

        with torch.no_grad():
            for feature in features:
                # prefix embedding을 다시 CLIP feature 공간으로 디코딩
                feature = self.caption_model.decode_prefix(feature.to(self.device))

                # 선택된 생성 방법으로 캡션 생성
                if use_beam_search:
                    # Beam search 사용 (더 안정적이고 품질 높은 결과)
                    generated_captions.append(generate_beam(self.caption_model, self.tokenizer, embed=feature)[0])
                else:
                    # Top-p sampling 사용 (더 다양한 결과)
                    generated_captions.append(generate2(self.caption_model, self.tokenizer, embed=feature))

        return generated_captions

    def train_model(self, json_path: str, output_dir: str, epochs: int = 5, batch_size: int = 4, lr: float = 1e-4):
        """
        모델 파인튜닝 함수
        새로운 데이터셋으로 기존 모델을 추가 학습

        Args:
            json_path: 훈련 데이터 JSON 파일 경로
            output_dir: 모델 저장 디렉토리
            epochs: 훈련 에포크 수
            batch_size: 배치 크기
            lr: 학습률
        """
        import os
        from torch.utils.data import Dataset, DataLoader
        from transformers import AdamW
        from tqdm import tqdm

        class CaptionDataset(Dataset):
            """
            캡션 생성을 위한 커스텀 데이터셋 클래스
            """

            def __init__(self, json_path, tokenizer, feature_dir, device):
                """
                데이터셋 초기화

                Args:
                    json_path: 데이터 JSON 파일 경로
                    tokenizer: 텍스트 토크나이저
                    feature_dir: CLIP feature 파일들이 저장된 디렉토리
                    device: 텐서가 위치할 디바이스
                """
                with open(json_path, 'r') as f:
                    self.data = json.load(f)
                self.tokenizer = tokenizer
                self.device = device
                self.feature_dir = feature_dir

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                """
                개별 데이터 아이템 반환

                Returns:
                    (clip_feat, input_ids, full_attention_mask):
                    CLIP 특징, 토큰화된 텍스트, 어텐션 마스크
                """
                item = self.data[idx]

                # CLIP feature 로드 (.npy 파일에서)
                npy_path = os.path.join(self.feature_dir, f"{idx}.npy")
                features = np.load(npy_path, allow_pickle=True).item()
                clip_feat = torch.tensor(features['output_text_latent'], dtype=torch.float32).to(self.device)

                # 텍스트 토큰화
                tokenized = self.tokenizer(
                    item['output_text'],
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=67  # 최대 시퀀스 길이
                )
                input_ids = tokenized.input_ids.squeeze(0).to(self.device)
                text_attention_mask = tokenized.attention_mask.squeeze(0).to(self.device)

                # 전체 어텐션 마스크 생성 (prefix + text)
                prefix_attention_mask = torch.ones(77, dtype=torch.long, device=self.device)  # prefix_length = 77
                full_attention_mask = torch.cat((prefix_attention_mask, text_attention_mask), dim=0)

                return clip_feat, input_ids, full_attention_mask

        # 데이터셋 및 데이터로더 준비
        print("Preparing dataset...")
        feature_dir = "/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/features2_dict/train"
        dataset = CaptionDataset(json_path, self.tokenizer, feature_dir, self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 훈련 준비
        print("Starting fine-tuning...")
        self.caption_model.train()  # 훈련 모드로 전환
        self.caption_model.requires_grad_(True)  # 기울기 계산 활성화

        # 옵티마이저 설정
        optimizer = AdamW(self.caption_model.parameters(), lr=lr)

        # 에포크별 훈련 루프
        for epoch in range(epochs):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            # 배치별 훈련
            for step, (clip_feats, input_ids, attn_mask) in enumerate(progress):
                optimizer.zero_grad()  # 기울기 초기화

                # Forward pass
                outputs = self.caption_model(
                    tokens=input_ids,
                    prefix=clip_feats,
                    mask=attn_mask,
                    labels=input_ids  # 자기회귀 언어모델이므로 입력과 라벨이 동일
                )

                # Backward pass
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # 손실 누적 및 진행상황 표시
                total_loss += loss.item()
                progress.set_postfix(loss=loss.item())

            # 에포크별 결과 출력
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

            # 디버깅용: state_dict 키 확인
            print(f"State dict keys at epoch {epoch + 1}:")
            for key in self.caption_model.state_dict().keys():
                print(f"  {key}")

            # 모델 저장
            model_save_path = os.path.join(output_dir, f"caption_decoder_epoch{epoch + 1}.pth")
            torch.save(self.caption_model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")


# 스크립트 실행 예시
if __name__ == "__main__":
    """
    메인 실행 부분
    명령줄 인자를 받아서 캡션 디코더를 초기화하고 파인튜닝을 수행

    사용법:
    python script.py --json_path /path/to/train.json --output_dir /path/to/save --pretrained_path /path/to/model.pth
    """
    import argparse
    import torch

    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='CLIP-GPT2 Caption Model Fine-tuning')
    parser.add_argument("--json_path", type=str,
                        default='/home/ivpl-d29/dataset/AvaMERG/train_finetune.json',
                        help='훈련 데이터 JSON 파일 경로')
    parser.add_argument("--output_dir", type=str,
                        default="/home/ivpl-d29/sichoi/Emo/unidiffuser/models",
                        help='모델 저장 디렉토리')
    parser.add_argument("--pretrained_path", type=str,
                        default='/home/ivpl-d29/sichoi/Emo/unidiffuser/models/caption_decoder.pth',
                        help='사전 훈련된 모델 파일 경로')
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument("--epochs", type=int, default=5,
                        help='훈련 에포크 수')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='배치 크기')
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help='학습률')

    args = parser.parse_args()

    # 캡션 디코더 초기화
    print(f"Initializing CaptionDecoder on {args.device}...")
    caption_decoder = CaptionDecoder(
        device=args.device,
        pretrained_path=args.pretrained_path
    )

    # 모델 파인튜닝 실행
    print("Starting model fine-tuning...")
    caption_decoder.train_model(
        json_path=args.json_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate
    )

    print("Fine-tuning completed!")

    """
    데이터셋 형식 설명:

    json_path에 지정된 파일은 다음과 같은 형식이어야 함:
    [
      {
        "input_image": "dia19900utt0_55.png",           # 입력 이미지 파일명
        "input_text": "I just can't seem to manage...",  # 입력 텍스트 (화자의 말)
        "output_image": "dia19900utt1_51.png",          # 출력 이미지 파일명
        "output_text": "That sounds really tough...",    # 출력 텍스트 (응답자의 말) - 이것이 생성 타겟
        "chain_of_empathy": {                            # 공감 체인 정보 (메타데이터)
          "speaker_emotion": "anxious",                  # 화자의 감정
          "event_scenario": "Struggling with work pressure", # 상황 설명
          "emotion_cause": "High workload and stress",   # 감정의 원인
          "goal_to_response": "Seek understanding and support" # 응답 목표
        }
      },
      ...
    ]

    특징:
    1. 이 모델은 감정적 대화 상황에서 공감적 응답을 생성하는 것이 목적
    2. input_text는 문제나 고민을 표현하는 내용
    3. output_text는 공감적이고 지지적인 응답
    4. CLIP feature는 별도의 .npy 파일로 저장되어 있어야 함
    5. chain_of_empathy는 훈련 시 직접 사용되지 않지만 데이터의 맥락 이해에 도움

    모델 아키텍처 요약:

    1. Input: CLIP Visual/Text Features (77 x 768)
    2. Prefix Encoding: Linear layer (optional hidden_dim)
    3. Prefix Decoding: Linear layer back to 768
    4. Concatenation: [Prefix Features | Text Embeddings]
    5. GPT-2 Processing: Autoregressive text generation
    6. Output: Generated empathetic response text

    주요 특징들:
    - Prefix-based conditioning: CLIP features를 prefix로 사용해 GPT-2 생성을 조건화
    - Beam Search: 고품질 텍스트 생성을 위한 탐색 알고리즘
    - Fine-tuning capability: 새로운 도메인 데이터로 추가 학습 가능
    - Empathetic response: 감정적 맥락을 이해하고 공감적 응답 생성
    """