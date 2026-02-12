import subprocess
import sys
def install_ffmpeg():
    print("Starting FFmpeg installation...")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("FFmpeg installation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during FFmpeg installation: {e}")

    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", 
            "ffmpeg.tar.xz"
        ])
        subprocess.check_call(["tar", "-xf", "ffmpeg.tar.xz"])

        result = subprocess.run(
            ["find", "/tmp", "-name","ffmpeg-*","-type","f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip().split('\n')[0]

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("FFmpeg binary installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading or installing the FFmpeg binary: {e}")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg verification failed: {e}")
        return False
