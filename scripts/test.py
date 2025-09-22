import librosa
import numpy as np

# 1. 다운로드된 파일의 정확한 이름을 여기에 넣어주세요.
#    (파일명이 길고 복잡하니 복사해서 붙여넣는 것을 추천합니다!)
audio_path = '[MR 제거] 아이유 (IU) - \'겨울잠\'.wav' 

try:
    print(f"'{audio_path}' 파일의 음색 분석을 시작합니다...")
    
    # 2. 오디오 파일 불러오기
    y, sr = librosa.load(audio_path)

    # 3. MFCCs 추출 (음색 특징 추출)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 4. 파일 전체를 대표하는 평균 MFCCs (음성 지문) 계산
    mfccs_mean = np.mean(mfccs, axis=1)

    # 5. 결과 확인
    print("-" * 30)
    print("음성 지문(MFCCs 평균값) 추출 완료!")
    print(mfccs_mean)

except Exception as e:
    print(f"오류 발생: {e}")