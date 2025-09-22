import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob

def get_advanced_features(file_path):
    """오디오 파일에서 특징 추출 (이전과 동일)"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        scaler = StandardScaler()
        return scaler.fit_transform(features.T)
    except Exception as e:
        print(f"'{os.path.basename(file_path)}' 처리 중 오류: {e}")
        return None

# --- 1단계: UBM (Universal Background Model) 학습 (이전과 동일) ---
print("UBM 학습을 시작합니다...")
ubm_training_dirs = ["iu_songs", "younha_songs", "sungsikyung_songs"]
ubm_features = []
for singer_dir in ubm_training_dirs:
    train_files = glob.glob(os.path.join(singer_dir, '*.wav'))
    for file_path in train_files:
        features = get_advanced_features(file_path)
        if features is not None:
            ubm_features.append(features)

if not ubm_features:
    print("오류: UBM 학습을 위한 데이터가 없습니다.")
    exit() # 데이터 없으면 종료

ubm_features_pooled = np.vstack(ubm_features)
UBM = GaussianMixture(n_components=64, covariance_type='diag', random_state=0).fit(ubm_features_pooled)
print("UBM 학습 완료!")


# --- i-vector 추출을 위한 함수 (이전과 동일) ---
def extract_simplified_ivector(features, ubm):
    adapted_gmm = GaussianMixture(n_components=ubm.n_components, covariance_type='diag', 
                                  means_init=ubm.means_, weights_init=ubm.weights_, 
                                  precisions_init=ubm.precisions_, random_state=0)
    adapted_gmm.fit(features)
    return adapted_gmm.means_.flatten()


# --- 설정 ---
# 1. 분석하고 싶은 가수의 노래들이 담긴 폴더
SINGER_NAME = "아이유"
SINGER_AUDIO_DIR = "iu_songs"

# 2. 비교할 사용자의 목소리 파일 (파일 1개)
USER_VOICE_PATH = "윤하 - 혜성 MR제거 [MR Removed].wav" # 여기에 본인이 녹음한 파일명을 넣으세요.
# ----------------


# --- 2단계: 가수 i-vector 추출 (폴더 내 모든 파일 사용) ---
print(f"\n'{SINGER_NAME}'의 통합 i-vector를 추출합니다...")
singer_files = glob.glob(os.path.join(SINGER_AUDIO_DIR, '*.wav'))
singer_features_list = [get_advanced_features(f) for f in singer_files if get_advanced_features(f) is not None]
if not singer_features_list:
    print(f"오류: '{SINGER_AUDIO_DIR}'에서 특징을 추출할 수 없습니다.")
    exit()

singer_features_pooled = np.vstack(singer_features_list)
ivector_singer = extract_simplified_ivector(singer_features_pooled, UBM)
print("가수 i-vector 추출 완료!")


# --- 3단계: 사용자 i-vector 추출 (파일 1개 사용) ---
print(f"'{os.path.basename(USER_VOICE_PATH)}' 파일의 i-vector를 추출합니다...")
user_features = get_advanced_features(USER_VOICE_PATH)

if user_features is None:
    print(f"오류: '{USER_VOICE_PATH}' 파일 처리 실패.")
    exit()

ivector_user = extract_simplified_ivector(user_features, UBM)
print("사용자 i-vector 추출 완료!")


# --- 4단계: 최종 비교 ---
similarity = cosine_similarity(ivector_singer.reshape(1, -1), ivector_user.reshape(1, -1))

print("\n--- i-vector 기반 최종 유사도 비교 ---")
print(f"학습된 가수 모델({SINGER_NAME})과 사용자 목소리({os.path.basename(USER_VOICE_PATH)})의 유사도:")
print(f"==> {similarity[0][0] * 100:.2f}%")