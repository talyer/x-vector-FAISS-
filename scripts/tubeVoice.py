import yt_dlp
import os

# 1. 다운로드할 유튜브 주소
url = 'https://www.youtube.com/watch?v=5JE2bl2yv1Y'
# 2. yt-dlp 옵션 설정
ydl_opts = {
    # 'bestaudio' -> 최고 음질의 오디오를 선택
    'format': 'bestaudio/best',
    
    # 파일명을 '영상제목.wav' 형식으로 저장하도록 설정
    # outtmpl의 확장자를 wav로 지정하면 후처리 과정에서 자동으로 변환됨
    'outtmpl': '%(title)s',
    
    # 후처리(postprocessors) 설정: 다운로드 후 오디오만 wav로 추출
    'postprocessors': [{
        'key': 'FFmpegExtractAudio', # FFmpeg를 사용해 오디오 추출
        'preferredcodec': 'wav',     # 선호하는 코덱을 wav로 설정
    }],
}

print(f"'{url}' 영상의 오디오 다운로드를 시작합니다...")

try:
    # 3. yt-dlp 객체를 생성하고 다운로드 실행
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    print("다운로드 및 .wav 변환이 완료되었습니다!")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")