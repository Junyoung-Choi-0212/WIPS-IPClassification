from transformers import AutoConfig, AutoModelForSequenceClassification

import torch

# 가능한 레이어 경로 후보
CANDIDATE_PATHS = [
    ["encoder", "layer"],               # BERT, RoBERTa
    ["model", "layers"],                # GPT, LLaMA, Falcon
    ["encoder", "block"],               # T5 encoder
    ["decoder", "block"],               # T5 decoder
    ["transformer", "h"],               # GPT-J, GPT-NeoX
    ["model", "decoder", "layers"],     # OPT, Meta LLaMA
    ["model", "blocks"],                # MPT
]

# [model_name] 모델을 불러와 다운스케일링을 진행한 다음 다운스케일링 된 모델을 반환하는 함수
def create_downscaled_model(hf_token, model_name, tokenizer, labels_list, layer_usage):
    config = AutoConfig.from_pretrained(model_name) # 모델의 config 불러오기
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = len(labels_list) # config를 불러와서 config = config로 지정 시 에러 발생하여 분리
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token, config = config) # 기본 모델 불러오기
    
    total_layers = config.num_hidden_layers # 모델의 레이어 수 확인
    downscaled_layers = max(1, int(total_layers * (layer_usage / 100))) # 사용자가 선택한 비율로 계산, max(a, b)는 두 값 중 큰 값을 선택(최소 1개의 레이어 보장)
    
    print(f"[{model_name}] 모델의 전체 레이어 수 : [{total_layers}], 사용할 레이어 수 : [{downscaled_layers}], 사용률 : [{layer_usage}%]")
    
    layers = None
    for path in CANDIDATE_PATHS:
        obj = model
        try:
            for p in path:
                obj = getattr(obj, p)
            if isinstance(obj, torch.nn.ModuleList):
                layers = obj
                layer_path = path
                break
        except AttributeError:
            continue

    if layers is None: # 경로 후보에서 찾지 못한 경우 레이어 수를 num_hidden_layers에 할당
        print(f"Can't find layer stack for [{model_name}], change config according to usage...")
        model.config.num_hidden_layers = downscaled_layers
    else:
        # 전체 레이어 중에서 중앙 레이어를 제외하고,
        # 하위(lower_n)와 상위(upper_m) 레이어만 선택하여 새 레이어 스택 구성
        # 사용할 레이어의 갯수가 홀수라면, 데이터 인식에 초점을 두도록 하위 레이어를 하나 더 많게 조절
        if downscaled_layers % 2 == 0:
            lower_n = int(downscaled_layers / 2)
        else:
            lower_n = int((downscaled_layers / 2) + 1)
        upper_m = int(downscaled_layers / 2)
        print(f"Layer downscaling : lower layer count : [{lower_n}], upper layer count : [{upper_m}]")
        selected_layers = layers[:lower_n] + layers[-upper_m:]
        new_layers = torch.nn.ModuleList(selected_layers) # 새 레이어 스택 교체

        # 모델에 반영
        obj_ref = model
        for p in layer_path[:-1]:
            obj_ref = getattr(obj_ref, p)
        setattr(obj_ref, layer_path[-1], new_layers)

        model.config.num_hidden_layers = len(new_layers) # config 갱신
        
    return model