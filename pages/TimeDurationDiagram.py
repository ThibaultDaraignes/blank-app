################################## Time-Duration Diagram #################################

import streamlit as st
import numpy as np 
import pandas as pd 
import datetime
import librosa
import soundfile as sf
import IPython.display as ipd
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6


def time_duration_visualization(S, hop_length_selection, start, duration, tick_spacing_selection):
    n_frames = S.shape[1]
    frame_indices = np.arange(n_frames)
    times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hop_length_selection)
    times = times + start
        
    duration = times[-1] # time of the last frame of the array, so approximatively the total duration of the signal

    #color map
    cmap = libfmp.b.compressed_gray_cmap(alpha=-5000)

    ####################### TIME-DURATION DIAGRAM + ENERGETIC PROFILE ######################

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.3)

    # ===================== MATRIX =====================
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(S, origin='upper', cmap=cmap, extent=[times[0], times[-1], times[-1], times[0]], aspect='auto')

    fig.colorbar(im, ax=ax, label='Similarity', fraction=0.046, pad=0.04)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    ticks = np.arange(start, start + duration - start, tick_spacing_selection) #change the ticks hop
    tick_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('Time (min:sec)')
    ax.set_ylabel('Duration (min:sec)')

    ax.grid(False)

    # ===================== DISPLAY : ENERGETIC PROFILE =====================
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(energy_per_line, np.arange(len(energy_per_line)), color='black')

    ax2.set_title("Energy profile")
    ax2.set_xlabel("Energy")

    ax2.invert_yaxis() 
    ax2.set_yticks([]) 
    ax2.grid(False)

    plt.tight_layout()
    plt.show()

    return st.pyplot(fig, width="content")


#================================= main program ===================================#

# title of the page
st.title("Time-Duration Diagram")
st.space("small")
st.text("""This page allows you to compute a time-duration diagram from a self-similarity matrix. 
        
First, select an audio file with the starting time and the duration you want (in sec) (example : start = 0 and duration = 10). The program display the waveform of your selection. Then, select the basic parameters of the FFT (hop_length and window) in order to plot it.

Now you can plot a self-similarity matrix based on the concordance as similairty measure : the concordance tells how much energy is shared between two adjacent frames. For a better display, choose the appropriate ticks spacing (depends on the duration of your audio sample).

Finally, you can compute a time-duration diagram, a new representation which apply an affine transformation to your self-similarity matrix. The y-axis counts the duration of the events instanciated in your signal in the audio when they are new and the x-axis is still the timeline.
        
There is also a plot of the energetic profile of the time-duration diagram, which computes the sum of all the lines of the diagram.""")

st.space("small")
# select an audio file
# if audio file selected, make computations
st.subheader("Audio sample")
audio_file = st.file_uploader("Choose an audio file", type="audio")
if audio_file:
    st.write("filename:", audio_file.name)
    with NamedTemporaryFile() as f:
        f.write(audio_file.read())
        wave, sr = librosa.load(f.name, sr=None)
    
        times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
        ts = pd.Series(wave, index=times, name=str(audio_file))
    #st.write(ts)

        start = st.number_input("Enter the starting time", min_value=0.0, max_value=times.to_list()[-1])
        duration = st.number_input("Enter the duration", min_value=0.1, max_value=times.to_list()[-1]-start)
        end = start + duration

        wave2, sr = librosa.load(f.name, sr=sr, offset=start, duration=duration)
    # st.audio(audio_file, start_time=start, end_time=end)

    st.html(ipd.Audio(wave2, rate=sr)._repr_html_())

    fig, ax = plt.subplots(figsize=(6, 6))
    librosa.display.waveshow(wave2, sr=sr, max_points=20000, axis='m')
    st.pyplot(fig, width='content')

    print(start, duration, end)

    st.space("medium")
    # fft spectrum
    st.subheader("Spectrum (short-term fourier transform)")
    hop_length_selection = st.selectbox("hop length for FFT", [512, 1024, 2048], index=0) #default = 512
    window_selection = st.selectbox("window for FFT", ["blackmanharris","blackman","hann", "hamming"], index=3) #default = hamming
    Xfft = np.abs(librosa.stft(wave2, hop_length=hop_length_selection, window=window_selection))
    Xfft_db = librosa.amplitude_to_db(Xfft, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 6))
    librosa.display.specshow(Xfft_db, sr=sr, x_axis="time", y_axis="log", cmap='grey')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(fig, width="content")


    st.space("medium")
    st.subheader("Time-Duration Diagram")
    #compute time-duration diagram
    n_frames = Xfft.shape[1] ### frame ==> event 
    S = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        for j in range(n_frames):
            if j >= i:
                S[i, j] = np.dot((Xfft[:, j]), Xfft[:, j-i])
            else:
                S[i, j] = 0

    # Ensure values are in proper range [0, 1] for similarity, with a percentil-based normalization 
    vmin, vmax = np.percentile(S, [5, 95])
    S = np.clip((S - vmin) / (vmax - vmin), 0, 1)

    #contrast enhancement
    S = (np.clip(S, 0.5, 1) - 0.5)*2

    tick_spacing_selection = st.selectbox("Select a tick spacing (in sec) for the display of the time-duration diagram", 
                                          [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 60], index=0)

    st.space("medium")
    st.text("Now here is the time-duration diagram !")
    
    # display the time-duration diagram with the energetic profile (at the right)
    energy_per_line = np.sum(S, axis=1)
    t = np.arange(len(energy_per_line))
    time_duration_visualization(S, hop_length_selection, start, duration, tick_spacing_selection)

    st.space("small")
    st.subheader("Energetic profile of your time-duration diagram")

    # display separately the energetic profile
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(t, energy_per_line) #time on x-axis and energy on y-axis
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energetic profile of time-duration diagram")
    plt.grid()
    plt.show()
    ax.grid(False, axis='x')
    ax.grid(False, axis='y')
    st.pyplot(fig, width="content")
