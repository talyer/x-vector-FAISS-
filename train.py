import librosa
import numpy as np
from speechbrain.inference import EncoderClassifier
from sklearn.preprocessing import StandardScaler
import os
import glob
import torch
import torchaudio
from torchaudio.transforms import Resample
from db_utils import init_db, upsert_xvector
from faiss_index import build_index
import joblib

# --- 특징 추출 함수 ---
def get_xvector(file_path, model):
    try:
        info = torchaudio.info(file_path)
        sr = info.sample_rate
        total_frames = info.num_frames
        target_sr = 16000
        chunk_sec = 10  # 10초 단위로 처리
        max_sec = 60    # 최대 60초까지만 사용해 메모리 사용 제한
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
            # 모노 다운믹스
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # 리샘플링
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

# --- 모델 불러오기 ---
print("미리 학습된 x-vector 모델을 불러옵니다...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("모델 로딩 완료!")

# --- DB 초기화 ---
init_db()

# --- 모델 디렉터리 생성 ---
os.makedirs("models", exist_ok=True)

# --- 각 가수별 x-vector 추출 및 DB 저장 ---
SINGER_DIRS = [
    os.path.join("data", "train", "iu_songs"),
    os.path.join("data", "train", "younha_songs"),
    os.path.join("data", "train", "sungsikyung_songs"),
] # 학습시킬 모든 가수 폴더
print("\n각 가수별 x-vector 추출 및 DB 저장을 시작합니다...")

total_saved = 0
singer_vectors = {}  # 가수별 x-vector 저장용

for singer_dir in SINGER_DIRS:
    singer_name = os.path.basename(singer_dir).replace("_songs", "")
    print(f"--> '{singer_name}' 학습 중...")
    
    singer_files = glob.glob(os.path.join(singer_dir, '*.wav'))

    if not singer_files:
        print(f"'{singer_dir}' 폴더에 파일이 없습니다. 건너뜁니다.")
        continue

    saved_count = 0
    singer_xvectors = []  # 해당 가수의 모든 x-vector
    
    for file_path in singer_files:
        xvector = get_xvector(file_path, classifier)
        if xvector is None:
            continue
        upsert_xvector(singer=singer_name, file_path=file_path, vector=xvector)
        singer_xvectors.append(xvector)
        saved_count += 1
    
    # 가수별 평균 x-vector 계산 및 저장
    if singer_xvectors:
        avg_xvector = np.mean(np.stack(singer_xvectors, axis=0), axis=0).astype(np.float32)
        singer_vectors[singer_name] = avg_xvector
        
        # models 디렉터리에 가수별 x-vector 저장
        model_path = os.path.join("models", f"{singer_name}_xvector.pkl")
        joblib.dump(avg_xvector, model_path)
        print(f"  '{singer_name}' 평균 x-vector 저장: {model_path}")
    
    print(f"'{singer_name}' 저장 완료: {saved_count}개 x-vector")
    total_saved += saved_count

if total_saved == 0:
    print("\nDB에 저장된 x-vector가 없습니다. 학습용 .wav 파일을 각 가수 폴더에 넣은 뒤 다시 실행하세요.")
else:
    print("\nFAISS 인덱스를 생성합니다...")
    index_path, n, dim = build_index()
    print(f"FAISS 인덱스 생성 완료: {index_path} (vectors={n}, dim={dim})")
    
    # 가수별 x-vector 요약 정보 저장
    if singer_vectors:
        summary_path = os.path.join("models", "singer_vectors_summary.pkl")
        joblib.dump(singer_vectors, summary_path)
        print(f"가수별 x-vector 요약 저장: {summary_path}")

print("\n작업이 완료되었습니다.")