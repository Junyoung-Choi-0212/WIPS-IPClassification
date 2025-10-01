import pandas as pd

class SlidingWindowChunker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def chunk_text(self, text, max_length=512, stride=50):
        tokens = self.tokenizer.encode(text, add_special_tokens=False) # 텍스트 토큰화
        chunks = [] # 잘린 텍스트 저장용 리스트
        start = 0 # 토큰을 잘라낼 시작 위치

        while start < len(tokens):
            end = min(start + max_length, len(tokens)) # start부터 max_length만큼 자르되, 텍스트 끝(len(tokens))을 넘지 않도록 자름
            chunk = self.tokenizer.decode(tokens[start:end], skip_special_tokens=True) # 잘라낸 토큰을 텍스트로 디코딩
            chunks.append(chunk) # 리스트에 추가

            print(f"[CHUNK] start={start}, end={end}, length={end-start}, text={chunk[:80]}...")

            if end == len(tokens): # 텍스트를 끝까지 잘랐다면 종료
                break

            start += max_length - stride # 다음 chunk 시작점 이동(문맥 유지를 위한 stride가 있어 일부 토큰이 중복됨)

        return chunks

    def create_chunked_dataset(self, df, max_length=512, stride=50):
        chunked_rows = []

        for _, row in df.iterrows():
            chunks = self.chunk_text(row["text"], max_length, stride) # dataframe의 각 row를 순회해 텍스트 조각 리스트를 저장

            for chunk in chunks: # 텍스트 조각을 새로운 row로 저장
                chunk_row = {"text": chunk, "patent_id": row["patent_id"]} 
                
                if "label" in row:
                    chunk_row["label"] = row["label"]
                chunked_rows.append(chunk_row)

        return pd.DataFrame(chunked_rows)