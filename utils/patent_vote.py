TOP_N = 3  # 각 chunk에서 예측 결과 상위 n개 라벨의 confidence만 합산

# patent_id(출원번호) 별 chunk의 예측 확률 중 상위 [TOP_N]개를 합산한 후 정규화를 통한 신뢰도 계산 
def patent_soft_voting(dataframe, probs, id2label):
    patent_results = []

    # 특허 단위로 그룹화
    for patent_id, group in dataframe.groupby('patent_id'):
        indices = group.index.tolist()
        group = group.copy()

        if len(indices) > 0:
            # 라벨 별 confidence 합산 저장
            label_conf_dict = {label: 0.0 for label in id2label.values()}
            
            print(f"\n [patent_id: {patent_id}]")
            
            for chunk_idx, idx in enumerate(indices):
                chunk_prob = probs[idx]
                
                # top-n 라벨 인덱스 (내림차순 정렬)
                top_indices = chunk_prob.argsort()[-TOP_N:][::-1]
                
                print(f"\n[Chunk {chunk_idx}] Top-{TOP_N} 예측:")
                for rank, label_idx in enumerate(top_indices, 1):
                    label = id2label[label_idx]
                    conf = float(chunk_prob[label_idx])

                    # 노이즈 제거
                    if conf < 0.3:
                        print(f"  {rank}. {label} (conf={conf:.4f}) >> 노이즈 제거됨")
                        continue

                    print(f"  {rank}. {label} (conf={conf:.4f})")
                    
                    # confidence 누적
                    label_conf_dict[label] += conf
                
            # 최종 결과: confidence 합이 가장 큰 라벨(soft voting) 선택
            pred_label = max(label_conf_dict, key=label_conf_dict.get)
            
            # 신뢰도 합 정규화
            total_conf = sum(label_conf_dict.values())
            if total_conf == 0:
                n_labels = len(label_conf_dict)
                if n_labels > 0:
                    label_conf_norm = {label: 1 / n_labels for label in label_conf_dict}
                else:
                    label_conf_norm = {}
            else:
                label_conf_norm = {label: conf / total_conf for label, conf in label_conf_dict.items()}
            
            pred_conf = round(label_conf_norm[pred_label], 4)

            print("\nChunk 별 confidence 합산 결과:")
            for label, conf_sum in sorted(label_conf_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"   {label}: {round(conf_sum,4)}")

            print(f"=> 최종 예측: {pred_label} (confidence 총합: {round(sorted(label_conf_dict.items(), key=lambda x: x[1], reverse=True)[0][1], 4)}, 신뢰도(confidence 정규화): {pred_conf})\n")

            patent_results.append({
                "출원번호": patent_id,
                "예측분류": pred_label,
                "신뢰도": pred_conf
            })
    
    return patent_results