import os
import torch
import numpy as np
from speechbrain.inference import EncoderClassifier
import torchaudio
from torchaudio.transforms import Resample
from faiss_index import search_top_k
from db_utils import get_singer_by_id
import glob

# --- 특징 추출 함수 ---
def get_xvector(file_path, model):
    try:
        info = torchaudio.info(file_path)
        sr = info.sample_rate
        total_frames = info.num_frames
        target_sr = 16000
        chunk_sec = 10
        max_sec = 60
        frames_per_chunk = int(sr * chunk_sec)
        max_frames = int(sr * max_sec)

        resampler = None if sr == target_sr else Resample(orig_freq=sr, new_freq=target_sr)

        embeddings = []
        frame_offset = 0
        processed = 0
        while frame_offset < total_frames and processed < max_frames:
            remaining = min(total_frames - frame_offset, max_frames - processed)
            num_frames = min(frames_per_chunk, remaining)
            if num_frames <= 0:
                break

            waveform, _ = torchaudio.load(file_path, frame_offset=frame_offset, num_frames=num_frames)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if resampler is not None:
                waveform = resampler(waveform)
            waveform = waveform.to(torch.float32)

            with torch.no_grad():
                emb = model.encode_batch(waveform)
            emb_np = emb.squeeze().cpu().numpy()
            embeddings.append(emb_np)

            frame_offset += num_frames
            processed += num_frames

        if not embeddings:
            return None
        return np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    except Exception as e:
        print(f"'{os.path.basename(file_path)}' 처리 중 오류: {e}")
        return None

# --- 1. 저장된 모델들 불러오기 ---
print("미리 학습된 x-vector 모델을 불러옵니다...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("모델 로딩 완료!")

print("\nFAISS 인덱스를 사용하여 검색합니다...")

# --- 2. 사용자 목소리 분석 ---
# 사용자 전용 디렉터리 생성 및 최신 파일 자동 선택
USER_DIR = os.path.join("data", "user", "user_voices")
os.makedirs(USER_DIR, exist_ok=True)

user_files = sorted(glob.glob(os.path.join(USER_DIR, "*.wav")), key=os.path.getmtime)
if not user_files:
    print(f"\n'{USER_DIR}' 폴더에 .wav 녹음 파일이 없습니다. 파일을 넣고 다시 실행하세요.")
    raise SystemExit(0)

USER_VOICE_PATH = user_files[-1]
print(f"\n사용자 목소리 '{USER_VOICE_PATH}'를 분석합니다...")
user_xvector = get_xvector(USER_VOICE_PATH, classifier)

if user_xvector is not None:
    # --- 3. FAISS Top-K 검색 ---
    scores, ids = search_top_k(user_xvector, k=5)
    
    # id → singer 매핑 및 집계
    singer_to_best = {}
    for score, row_id in zip(scores, ids):
        if row_id == -1:
            continue
        singer = get_singer_by_id(int(row_id))
        if not singer:
            continue
        pct = float(score) * 100.0
        if singer not in singer_to_best or singer_to_best[singer] < pct:
            singer_to_best[singer] = pct

    if singer_to_best:
        best_match_singer = max(singer_to_best, key=singer_to_best.get)
        print("\n--- 최종 분석 결과 (FAISS/ECAPA) ---")
        for singer, pct in sorted(singer_to_best.items(), key=lambda x: x[1], reverse=True):
            print(f"'{singer}' 모델과의 유사도: {pct:.2f}%")
        print(f"\n==> 최종 판정: 가장 유사한 가수는 '{best_match_singer}' 입니다!")
    else:
        print("결과를 계산할 수 없습니다.")