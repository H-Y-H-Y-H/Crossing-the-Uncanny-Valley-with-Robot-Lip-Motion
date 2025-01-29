import librosa
import numpy as np
import matplotlib.pyplot as plt

import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def adjust_audio_volume_to_video(audio_path, video_frame_count, frame_rate=30):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Calculate the RMS value for each frame
    rms = librosa.feature.rms(y=y)[0]

    # Normalize the RMS values to 0-1
    normalized_rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

    # Calculate the audio duration and corresponding times for RMS values
    audio_duration = len(y) / sr
    audio_times = np.linspace(0, audio_duration, len(rms))

    # Video duration based on frame count and frame rate
    video_duration = video_frame_count / frame_rate

    # Generate video times based on the exact frame count
    video_times = np.linspace(0, video_duration, video_frame_count)

    # Interpolate RMS values to match video frame count
    interpolation_func = interp1d(audio_times, normalized_rms, kind='linear', bounds_error=False,
                                  fill_value=(normalized_rms[0], normalized_rms[-1]))
    normalized_volume_curve_video = interpolation_func(video_times)

    return video_times, normalized_volume_curve_video

def adjust_curve(values, threshold=0.2, strength=10):
    """
    Adjust values based on a threshold. Values above the threshold are moved closer to 1,
    and values below are moved closer to 0.

    :param values: The input array of values to adjust.
    :param threshold: The threshold for adjustment.
    :param strength: How strongly to adjust the values (higher = more adjustment).
    :return: Adjusted values.
    """
    # Piecewise linear transformation could be one approach, but here we're using
    # a smooth transition for a more natural effect.
    adjusted = np.zeros_like(values)
    for i, val in enumerate(values):
        if val > threshold:
            # Scale the value between threshold and 1
            adjusted[i] = 1 - np.power(1 - val, strength)
        else:
            # Scale the value between 0 and threshold
            adjusted[i] = np.power(val / threshold, strength) * threshold
    return adjusted

def moving_average(data, window_size):
    """Smooth data by calculating the moving average."""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

# # Example usage

for demo_id in range(20,21):
    frames_lmks = np.load('../../EMO_GPTDEMO/robot_data/data1001/synthesized/lmks/m_lmks_%d.npy'%demo_id)

    video_frame_count = len(frames_lmks)  # Use the frame count obtained from OpenCV

    audio_path =f'../../EMO_GPTDEMO/audio/emo/emo{demo_id}.mp3'

    frame_rate = 30  # Assuming 30 fps for your video

    video_times, normalized_volume_curve_video = adjust_audio_volume_to_video(audio_path, video_frame_count, frame_rate)


    # Example usage with a moving average
    window_size = 5  # Number of samples over which to average
    smoothed_volume_curve = moving_average(normalized_volume_curve_video, window_size)


    # Assuming normalized_volume_curve_video is your normalized volume data
    # Example usage:
    threshold = 0.05
    strength = 10  # Adjust this to control how aggressive the adjustment is

    # Adjust the volume curve based on the threshold
    adjusted_volume_curve = adjust_curve(smoothed_volume_curve, threshold, strength)
    np.savetxt(f'volume_tune/audio_factor{demo_id}.csv', adjusted_volume_curve)
    adjusted_volume_curve = adjusted_volume_curve*0.8+0.2
    # Plot
    plt.plot(video_times, smoothed_volume_curve)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Volume')
    plt.title('Normalized Volume Curve Aligned with Video Frame Count')
    plt.show()
# quit()
import contextlib,wave
from pydub import AudioSegment


def plot_audio_waveform(file_path, frame_rate=30):
    """
    Plots the waveform of an audio file with time on the x-axis and normalized amplitude.
    Supports both WAV and MP3 formats.
    frame_rate: Frame rate of the audio file in FPS (default is 30fps).
    """
    # Convert mp3 file to wav if necessary
    if file_path.endswith('.mp3'):
        # Convert mp3 to wav
        sound = AudioSegment.from_mp3(file_path)
        file_path = file_path.replace('.mp3', '.wav')
        sound.export(file_path, format="wav")

    # Open the audio file as a waveform
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.readframes(-1)
        sound_info = np.frombuffer(frames, dtype=np.int16)
        nframes = f.getnframes()
        framerate = f.getframerate()
        nchannels = f.getnchannels()

    # If the audio is stereo, convert it to mono
    if nchannels == 2:
        sound_info = np.mean(sound_info.reshape(-1, 2), axis=1)

    # Calculate window size and mean values for each window
    sound_info = np.abs(sound_info)
    window_size = int(framerate / frame_rate)
    mean_values = [np.mean(sound_info[i:i + window_size]) for i in range(0, len(sound_info), window_size)]

    duration = len(mean_values) / frame_rate
    # Normalize the sound data
    mean_values = (mean_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))


    # Create a time array in seconds
    time = np.linspace(0, duration, num=len(mean_values))
    mean_values = np.clip(mean_values,0,0.8)
    mean_values = (mean_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))

    cmds = 1-mean_values
    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, cmds)
    plt.title('Audio Waveform from MP4 with Mean Amplitude in Windows')
    plt.ylabel('Mean Normalized Amplitude')
    plt.xlabel('Time (seconds)')
    plt.show()

    np.savetxt(f'wav_bl_cmds/audio_v{demo_id}.csv', cmds)
# Example usage (you would replace 'audio_file.wav' with your file path)
for demo_id in range(20,21):
    audio_path =f'../../EMO_GPTDEMO/audio/emo/emo{demo_id}.mp3'
    plot_audio_waveform(audio_path)

# Note: This script expects the audio file to be present in the same directory.
# For MP3 files, it will create a temporary WAV file in the same directory.
