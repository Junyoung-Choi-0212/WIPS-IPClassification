TOP_N = 3  # 각 chunk에서 예측 결과 상위 n개 라벨의 confidence만 합산

# patent_id(출원번호) 별 chunk의 예측 확률 중 상위 [TOP_N]개를 합산한 후 확률이 가장 높은 분류를 선택
def patent_soft_voting(dataframe, probs, id2label):
    print("[inference dataset]")
    print(dataframe.head(5))
    
    patent_results = []
    for patent_id, group in dataframe.groupby('patent_id'): # 특허 단위로 그룹화
        indices = group.index.tolist()
        group = group.copy()
        
        merged_text = group["text"].str.cat(sep=" ") # 특허 id 기준 텍스트 통합

        if len(indices) > 0:
            label_conf_dict = {label: 0.0 for label in id2label.values()} # 라벨 별 confidence를 합산 저장할 딕셔너리
            
            print(f"\n [patent_id: {patent_id}]")
            
            for chunk_idx, idx in enumerate(indices):
                chunk_prob = probs[idx] # 현재 chunk의 예측 확률 가져오기
                top_indices = chunk_prob.argsort()[-TOP_N:][::-1] # top-n 라벨 인덱스 (내림차순 정렬)
                
                print(f"\n[Chunk {chunk_idx}] Top-{TOP_N} 예측:")
                for rank, label_idx in enumerate(top_indices, 1): # 설정한 TOP_N 변수에 맞게 반복하여 딕셔너리에 확률 저장
                    label = id2label[label_idx]
                    conf = float(chunk_prob[label_idx])

                    print(f"  {rank}. {label} (conf={conf:.4f})")
                    
                    label_conf_dict[label] += conf # confidence 누적(※ confidence 값은 최종 분류를 선택하기 위해서만 사용 ※)
                
            pred_label = max(label_conf_dict, key=label_conf_dict.get) # 최종 결과: confidence 합이 가장 큰 라벨(soft voting) 선택

            print("\nChunk 별 confidence 합산 결과:")
            for label, conf_sum in sorted(label_conf_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"   {label}: {round(conf_sum,4)}")

            print(f"=> 최종 예측: {pred_label}\n")

            patent_results.append({
                "patent_id": patent_id,
                "text": merged_text,
                "text_preview": merged_text[:100] + "..." if len(merged_text) > 100 else merged_text,
                "classification": pred_label
            })
    
    return patent_results