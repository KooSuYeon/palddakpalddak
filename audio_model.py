import os
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
import io
from pydub.utils import which

load_dotenv()

# 세팅
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

output_filename = "output_audio.mp3"

def generate_audio_from_text(text):
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    # API URL과 API 키를 설정합니다.
    url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    api_key = os.getenv("ELEVENLABS_API_KEY")

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 1,
            "style": 0.3,
            "use_speaker_boost": True
        }
    }


    response = requests.post(url, json=data, headers=headers, stream=True)

    if response.status_code == 200:
        audio_content = b""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_content += chunk
        return audio_content
    else:
        print(f"Failed to generate audio: {response.status_code}")
        return None

def save_audio(audio_content, output_filename):
    segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
    segment.export(output_filename, format="mp3")
    print(f"Success! Wrote audio to {output_filename}")

def play_audio(audio_content):
    segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
    play(segment)

if __name__ == "__main__":
    # 사용자로부터 텍스트를 입력받습니다.
    text1 = "로마는 1승 3무 1패(승점 6)로 21위에 자리했다. 올 시즌부터 UEL은 챔피언스리그(UCL)와 동일하게 본선 무대에 오른 36개 팀이 리그 페이즈에서 8경기(홈 4경기·원정 4경기)씩 치른다. 이후 상위 1~8위 팀은 16강에 직행하고, 9위에서 24위 팀은 플레이오프를 벌여 승자가 16강에 합류한다."
    text2= "Son Heung-min (Tottenham Hotspur) scored his fourth goal in the 2024-2025 UEFA Europa League (UEL). He scored his first goal in this season's European match and rejoiced with a click ceremony."

    
    # 음성을 생성하고 저장 및 재생합니다.
    audio_content = generate_audio_from_text(text1)

    if audio_content:
        play_audio(audio_content)
        save_audio(audio_content, output_filename)
