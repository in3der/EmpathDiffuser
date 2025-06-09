import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import default_data_collator
from transformers import EarlyStoppingCallback

data_collator = default_data_collator
es = EarlyStoppingCallback(early_stopping_patience=5)
import json
import argparse
from typing import Union, Optional
from collections import OrderedDict


# %% model initial
class ClipCaptionModel(nn.Module):
    """
    CLIP feature를 입력으로 받아 GPT-2를 이용해 캡션(텍스트)을 생성하는 모델.
    입력된 CLIP feature를 prefix로 사용하여 GPT-2 텍스트 생성을 조건화함.
    """

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # loss 계산 시 prefix에 해당하는 자리만큼 dummy token으로 padding
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        GPT-2에 prefix(CLIP feature)와 token(GPT 텍스트 입력)을 입력해 캡션을 생성하는 forward 함수
        : param tokens: (Tensor) [N x max_seq_len] eg. [4 X 33]
        : param prefix: (Tensor) [N x prefix_length x 768] eg. [4 x 77 x 768]
        : param mask: (Tensor) [N x (prefix_length + max_seq_len) x 768] eg. [4 x 110 x768]

        : attribute embedding_text: (Tensor) [N x max_seq_len x 768] eg. [4 x 33 x 768]
        : attribute embedding_cat: (Tensor) [N x (prefix_length + max_seq_len) x 768] eg. [4 x 110 x 768]
        """
        # 토큰을 GPT embedding 공간으로 매핑
        embedding_text = self.gpt.transformer.wte(tokens)
        # CLIP feature를 임베딩 공간으로 인코딩 및 디코딩 (중간 hidden_dim을 거칠 수 있음)
        hidden = self.encode_prefix(prefix)
        prefix = self.decode_prefix(hidden)
        # prefix와 텍스트 임베딩을 시퀀스 차원에서 이어붙임
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)

        if labels is not None:
            # loss를 계산할 때 prefix 자리에 dummy label을 추가 (무시되도록 0으로 채움)
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        # GPT-2에 embedding 입력
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        # hidden_dim이 설정되어 있으면 hidden도 반환 (prefix encoding 정보)
        if self.hidden_dim is not None:
            return out, hidden
        else:
            return out

    def encode_decode_prefix(self, prefix):
        # prefix를 encode하고 다시 decode해서 원래 공간으로 되돌림 (예: 학습 또는 검증 시 사용)
        return self.decode_prefix(self.encode_prefix(prefix))

    def __init__(self, prefix_length: int, hidden_dim=None):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        # GPT-2 tokenizer 초기화 및 EOS 토큰 추가
        base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        base_tokenizer.add_special_tokens(special_tokens_dict)
        # GPT-2 모델 초기화 (language modeling head 포함)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2', eos_token_id=base_tokenizer.eos_token_id)
        self.gpt.resize_token_embeddings(len(base_tokenizer))

        self.hidden_dim = hidden_dim
        # 입력 prefix가 hidden_dim을 거쳐 GPT 임베딩 차원(768)에 매핑되도록 설정
        self.encode_prefix = nn.Linear(768, hidden_dim) if hidden_dim is not None else nn.Identity()
        self.decode_prefix = nn.Linear(hidden_dim, 768) if hidden_dim is not None else nn.Identity()




def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
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
        prompt=None,  # 텍스트 프롬프트 (사용하지 않음)
        embed=None,  # GPT 입력 임베딩 (보통은 CLIP 임베딩에서 변환된 결과)
        entry_length=67,  # 생성할 최대 토큰 수
        temperature=1.0,  # softmax 온도 (작을수록 더 결정적인 선택)
        stop_token: str = '<|EOS|>',  # 종료 토큰
):
    """ Bean Search를 이용해 GPT based model로부터 text를 생성 """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]  # 종료 토큰의 인덱스를 찾음
    tokens = None  # 현재까지 생성된 토큰
    scores = None  # 각 beam의 점수
    device = next(model.parameters()).device  # 모델이 위치한 디바이스
    seq_lengths = torch.ones(beam_size, device=device)  # 각 시퀀스 길이 추적
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)  # 각 beam이 종료되었는지 여부

    with torch.no_grad():
        if embed is not None:
            # 미리 제공된 임베딩이 있으면 그것을 사용 (예: CLIP feature 기반)
            generated = embed
        else:
            # 텍스트 프롬프트를 토큰화하고 GPT embedding으로 변환
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        # pbar = tqdm(range(entry_length))
        # pbar.set_description("generating text ...")
        # 반복적으로 토큰을 생성
        for i in range(entry_length):
            # print(generated.shape)
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                # 첫 토큰 생성 시, top-k 후보로 beam 시작
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])  # beam_size만큼 복제
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens  # 처음 생성된 토큰 저장
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                # 그 이후의 step부터는 확장된 시퀀스를 평가
                logits[is_stopped] = -float(np.inf)  # 이미 종료된 beam은 확장하지 않음
                logits[is_stopped, 0] = 0  # 종료된 beam에 대해 [PAD] 처리를 위한 0 확률

                scores_sum = scores[:, None] + logits  # 누적 점수 계산
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]

                # top-k 후보 선택
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1] # 어떤 beam에서 나왔는지
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                # 시퀀스, 임베딩, 점수 갱신
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            # 새로 생성된 토큰을 임베딩하고 이어붙임
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)

            # 종료 토큰이 나왔는지 확인
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break  # 모든 beam이 종료되면 루프 중단

    # 점수를 길이로 나눠 normalize
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    # 토큰을 텍스트로 디코딩
    output_texts = [
        tokenizer.decode(output[: int(length)], skip_special_tokens=True)
        for output, length in zip(output_list, seq_lengths)
    ]
    # 점수 기준으로 정렬
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    model.train()  # 다시 학습 모드로 변경
    return output_texts  # 최종 생성된 텍스트 리스트 (score 기준 정렬됨)


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.0,
        stop_token: str = '<|EOS|>',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


class CaptionDecoder(object):
    def __init__(self, device, pretrained_path, hidden_dim=-1):
        # hidden_dim 기본값이 -1이면 None으로 설정 (모델 초기화 시 옵션)
        if hidden_dim < 0:
            hidden_dim = None
        # tokenizer initialize
        # GPT-2 tokenizer 사용, EOS(End of Sentence) 토큰을 특별 토큰으로 추가
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # model initialize # 모델 초기화
        feature_length = 77        # 입력 feature 길이 (clip feature 길이)
        # modelFile = "assets/caption_decoder/coco_v2_latest.pt"
        self.caption_model = ClipCaptionModel(feature_length, hidden_dim=hidden_dim)
        # print("Load Model...")
        ckpt = torch.load(pretrained_path, map_location='cpu')  # 사전 학습된 모델 가중치 불러오기
        state_dict = OrderedDict()
        # 저장된 state_dict 키 앞부분 'module.' 제거 (DistributedDataParallel 등에서 붙는 prefix)
        for k, v in ckpt.items():
            new_k = k[7:]
            state_dict[new_k] = v

        # 기존 state_dict 초기화
        state_dict = {}
        for k, v in ckpt.items():
            if k.startswith('module.'):
                k = k[7:]  # 'module.' 제거
            state_dict[k] = v

        # 3. 'nsformer' → 'gpt.transformer'로 prefix 교체
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('nsformer'):
                new_k = 'gpt.transformer' + k[
                                            len('nsformer'):]  # ex) nsformer.h.0.ln_1.weight → gpt.transformer.h.0.ln_1.weight
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v

        # 가중치 로드
        mk, uk = self.caption_model.load_state_dict(new_state_dict, strict=False)

        # 디버그 출력
        print('Missing keys (mk): ', mk)
        print('Unexpected keys (uk): ', uk)

        # 로드되지 않은 키들이 허용된 prefix를 가지는지 확인
        allowed_prefixes = ('clip', 'prefix', 'gpt', 'transformer', 'nsformer', 'encode_prefix', 'decode_prefix', 'lm_head')
        assert all([name.startswith(allowed_prefixes) for name in uk])

        # 모델을 평가 모드로 설정하고, device에 올림
        self.caption_model.eval()
        self.caption_model.to(device)
        self.caption_model.requires_grad_(False)
        self.device = device

        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('nsformer'):
        #         new_key = 'gpt.transformer' + k[len('nsformer'):]
        #         new_state_dict[new_key] = v
        #     else:
        #         new_state_dict[k] = v
        #
        # # 가중치 로드, strict=False로 일부 가중치 누락 허용
        # mk, uk = self.caption_model.load_state_dict(state_dict, strict=False)
        #
        #
        # #assert len(mk) == 0 # 로드하지 못한 가중치가 없어야 함
        # print('mk : ', mk)
        # print('uk : ', uk)
        # # assert all([name.startswith('clip') for name in uk])  # 250504 - weight 불러올 때 에러나서 주석처리함
        # # 대신에 추가  # 로드하지 못한 가중치 이름이 'clip', 'prefix', 'gpt', 'transformer' 중 하나로 시작하는지 확인
        # allowed_prefixes = ('clip', 'prefix', 'gpt', 'transformer', 'nsformer')
        # assert all([name.startswith(allowed_prefixes) for name in uk])
        #
        # self.caption_model.eval()
        # self.caption_model.to(device)
        # self.caption_model.requires_grad_(False)
        # self.device = device

    def encode_prefix(self, features): # 입력된 clip feature를 prefix embedding으로 인코딩
        return self.caption_model.encode_prefix(features)

    def generate_captions(self, features):  # the low dimension representation of clip feature
        """
        generate captions given features  clip feature의 저차원 표현(features)을 입력으로 받아 캡션(텍스트) 생성
        : param features : (tensor([B x L x D]))
        : return generated_text: (list([L]))
        """

        # generate config
        use_beam_search = True

        # 배치 단위로 feature를 쪼갬 (한 번에 한 feature씩 처리)
        features = torch.split(features, 1, dim=0)
        generated_captions = []
        with torch.no_grad():   # 추론 시 기울기 계산 비활성화
            for feature in features:
                # prefix embedding을 다시 clip feature 공간으로 디코딩
                feature = self.caption_model.decode_prefix(feature.to(self.device))  # back to the clip feature
                # 빔 서치 혹은 샘플링을 이용해 텍스트 생성
                if use_beam_search:
                    generated_captions.append(generate_beam(self.caption_model, self.tokenizer, embed=feature)[0])
                else:
                    generated_captions.append(generate2(self.caption_model, self.tokenizer, embed=feature))
        return generated_captions
