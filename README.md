## 프로젝트 개요

ECAPA-TDNN 기반 x-vector를 추출하여 SQLite(DB)에 저장하고, FAISS로 최근접 검색(코사인 유사도 등가의 inner product)해 화자/가수 유사도를 계산합니다.

주요 파일
- `train.py`: 학습 데이터(.wav)에서 x-vector 추출 → DB 저장 → FAISS 인덱스 생성
- `run_app.py`: 사용자 음성(.wav)에서 x-vector 추출 → FAISS로 검색 → 결과 출력
- `db_utils.py`: SQLite 유틸 (저장/조회)
- `faiss_index.py`: FAISS 인덱스 빌드/로드/검색
- `requirements.txt`: 의존성 목록
- `scripts/`: 보조 스크립트(`compare_voices.py`, `test.py`, `tubeVoice.py` 등)
- `models/`: 예전 평균 `.xvector` 파일(현재 파이프라인에서는 미사용)

디렉터리 구조(권장)
- `iu_songs/`, `younha_songs/`, `sungsikyung_songs/`: 각 가수의 학습용 `.wav`
- `user_voices/`: 사용자 녹음 `.wav` (최신 파일 자동 선택)

## 환경 준비

1) Python 3.10~3.13 권장(Windows)
2) 패키지 설치
```bash
pip install -r requirements.txt
```
3) FAISS가 Windows+pip 환경에서 문제 시, Conda 권장
```bash
conda install -c pytorch faiss-cpu
```

## 데이터 배치

- 학습 데이터(.wav)를 아래 폴더에 넣습니다.
  - `iu_songs/`
  - `younha_songs/`
  - `sungsikyung_songs/`
- 추가 가수 예: `taeyeon_songs/` 폴더를 만들고 `train.py`의 `SINGER_DIRS` 리스트에 추가
- 파일 형식: `.wav` 권장

## 학습(DB 저장) & 인덱스 생성

```bash
python train.py
```
- 각 `.wav`에서 x-vector를 추출해 `xvectors.sqlite3`에 저장합니다.
- 저장된 x-vector로 FAISS 인덱스를 생성합니다.
- 메모리 사용을 줄이기 위해 10초 청크, 최대 60초까지만 사용해 임베딩을 평균냅니다.

출력 예시
```
미리 학습된 x-vector 모델을 불러옵니다...
모델 로딩 완료!
각 가수별 x-vector 추출 및 DB 저장을 시작합니다...
--> 'iu' 저장 완료: 11개 x-vector
...
FAISS 인덱스를 생성합니다...
FAISS 인덱스 생성 완료: C:\\Users\\<USER>\\AppData\\Local\\Temp\\faiss.index (vectors=29, dim=192)
```

## 추론(사용자 음성 분석)

1) `user_voices/` 폴더에 사용자의 녹음 `.wav`를 넣습니다. 최신 파일이 자동 선택됩니다.
2) 실행
```bash
python run_app.py
```
3) 출력: FAISS Top-K 검색 후 가수별 최고 유사도를 정렬해 표시합니다.

## 자주 묻는 문제(FAQ)

- Q) 인덱스 저장 에러(Illegal byte sequence)가 납니다.
  - A) 인덱스는 시스템 TEMP(영문 경로)에 저장하도록 코드가 설정되어 있습니다. 환경변수 `TEMP`를 영문 경로로 지정해 사용하세요.

- Q) 메모리 부족(alloc_cpu) 오류가 납니다.
  - A) 이미 청크 처리(10초, 최대 60초)로 제한합니다. 그래도 부족하면 `train.py`/`run_app.py`의 `chunk_sec`를 5로 줄이고, `max_sec`도 30으로 낮추세요.

- Q) 모델 import 경고가 보입니다.
  - A) `speechbrain.inference.EncoderClassifier`를 사용합니다(코드 반영됨). 최초 실행 시 모델 다운로드로 시간이 소요될 수 있습니다.

- Q) OneDrive 경로에서 느립니다.
  - A) 가능하면 로컬 경로(예: `C:\\work\\TEST`)로 옮겨 실행하세요.

## 개발 메모

- 유사도: 코사인 유사도(벡터 L2 정규화 후 `IndexFlatIP`)
- DB 스키마: `xvectors(id, singer, file_path, dim, vector BLOB, created_at)`
- 인덱스 파일: 기본 `TEMP/faiss.index`

## 라이선스/모델 출처

- Speaker model: `speechbrain/spkrec-ecapa-voxceleb`
- 라이선스와 세부 내용은 각 라이브러리/모델 저장소를 참고하세요.


