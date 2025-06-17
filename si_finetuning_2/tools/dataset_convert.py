import json
import random
import os

def convert_dialogue_to_finetune_samples(input_json_path, train_output_path, test_output_path, test_ratio=0.2, seed=42):
    random.seed(seed)

    with open(input_json_path, 'r') as f:
        conversations = json.load(f)

    output_data = []

    for conv in conversations:
        conv_id = conv["conversation_id"]
        speaker_id = conv["speaker_profile"]["ID"]
        listener_id = conv["listener_profile"]["ID"]

        for turn in conv["turns"]:
            chain_of_empathy = turn["chain_of_empathy"]
            response = turn["response"]
            turn_id = int(turn["turn_id"])  # 문자열이면 int로 변환

            index = turn_id * 2  # input_image용 index 계산
            input_image = f"dia{conv_id}utt{index}_{speaker_id}.png"
            output_image = f"dia{conv_id}utt{index + 1}_{listener_id}.png"

            # turn["dialogue_history"] 내 speaker 발화 중 마지막 것만 input_text로 사용
            speaker_utterances = [h["utterance"] for h in turn["dialogue_history"] if h["role"] == "speaker"]
            if speaker_utterances:
                input_text = speaker_utterances[-1]

                sample = {
                    "input_image": input_image,
                    "input_text": input_text,
                    "output_image": output_image,
                    "output_text": response,
                    "chain_of_empathy": chain_of_empathy
                }
                output_data.append(sample)

    # 셔플 및 분할
    random.shuffle(output_data)
    test_size = int(len(output_data) * test_ratio)
    test_data = output_data[:test_size]
    train_data = output_data[test_size:]

    # 저장
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"총 샘플 수: {len(output_data)} → train: {len(train_data)}, test: {len(test_data)}")


# 실행
convert_dialogue_to_finetune_samples(
    input_json_path="/home/ivpl-d29/dataset/AvaMERG/train.json",
    train_output_path="/home/ivpl-d29/dataset/AvaMERG/train_finetune.json",
    test_output_path="/home/ivpl-d29/dataset/AvaMERG/test_finetune.json"
)
