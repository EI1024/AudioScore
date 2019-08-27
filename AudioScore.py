import wave
import os
import numpy as np

import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy

import librosa

import pickle
import struct

import sys
from PyQt5.QtWidgets import *

def con_wave(init_pitch, counter,new_x,voice_dict,log_cqt_power,inputsec):

    key = init_pitch
    cutlen = (inputsec / log_cqt_power.shape[1]) * counter * 44100 * 2

    if key%12 == 0: # C
        if key < 10:
            x_add = voice_dict['C4'][0:int(cutlen)]
        else:
            x_add = voice_dict['C5'][0:int(cutlen)]

    elif key%12 == 1: # CC
        if key < 10:
            x_add = voice_dict['CC4'][0:int(cutlen)]
        else:
            x_add = voice_dict['CC5'][0:int(cutlen)]

    elif key%12 == 2: # D
        if key < 20:
            x_add = voice_dict['D4'][0:int(cutlen)]
        else:
            x_add = voice_dict['D5'][0:int(cutlen)]

    elif key%12 == 3: # DD
        if key < 20:
            x_add = voice_dict['DD4'][0:int(cutlen)]
        else:
            x_add = voice_dict['DD5'][0:int(cutlen)]

    elif key%12 == 4: # E
        if key < 20:
            x_add = voice_dict['E4'][0:int(cutlen)]
        else:
            x_add = voice_dict['E5'][0:int(cutlen)]

    elif key%12 == 5: # F
        if key < 10:
            x_add = voice_dict['F3'][0:int(cutlen)]
        elif key < 20:
            x_add = voice_dict['F4'][0:int(cutlen)]
        else:
            x_add = voice_dict['F5'][0:int(cutlen)]

    elif key%12 == 6: # FF
        if key < 10:
            x_add = voice_dict['FF3'][0:int(cutlen)]
        elif key < 20:
            x_add = voice_dict['FF4'][0:int(cutlen)]
        else:
            x_add = voice_dict['FF5'][0:int(cutlen)]

    elif key%12 == 7: # G
        if key < 10:
            x_add = voice_dict['G3'][0:int(cutlen)]
        else:
            x_add = voice_dict['G4'][0:int(cutlen)]

    elif key%12 == 8: # GG
        if key < 10:
            x_add = voice_dict['GG3'][0:int(cutlen)]
        else:
            x_add = voice_dict['GG4'][0:int(cutlen)]

    elif key%12 == 9: # A
        if key < 10:
            x_add = voice_dict['A3'][0:int(cutlen)]
        else:
            x_add = voice_dict['A4'][0:int(cutlen)]

    elif key%12 == 10: # AA
        if key < 20:
            x_add = voice_dict['AA3'][0:int(cutlen)]
        else:
            x_add = voice_dict['AA4'][0:int(cutlen)]

    elif key%12 == 11: # B
        if key < 20:
            x_add = voice_dict['B3'][0:int(cutlen)]
        else:
            x_add = voice_dict['B4'][0:int(cutlen)]

    else:
        x_add = voice_dict['NoVoice'][0:int(cutlen)]


    new_x = np.concatenate([new_x, x_add], axis=0)
    return new_x

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.button = QPushButton('変換')
        self.button.clicked.connect(self.output)
        self.inputText = QLineEdit()
        self.inputText.setText("")

        textLayout = QHBoxLayout()
        textLayout.addWidget(self.inputText)

        layout = QVBoxLayout()
        layout.addLayout(textLayout)
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.setWindowTitle("AudioScore")

    def output(self):

    ########################################
        filename = self.inputText.text()
        base = os.getcwd()+ '/'
        with open(base + 'Voice.pickle', mode='rb') as f:
            voice_dict = pickle.load(f)

        y, sr = librosa.load(base + filename +".wav")
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        y = y_harmonic
        C = librosa.cqt(y, sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        wavfile = wave.open(base + filename + ".wav" , "rb" )
        sampling_rate = wavfile.getframerate()
        channel_num = wavfile.getnchannels()
        sample_size = wavfile.getsampwidth()
        frame_num = wavfile.getnframes()
        inputsec = frame_num/sampling_rate
        wavfile.close()

        log_cqt_power = librosa.amplitude_to_db(np.abs(C), ref=np.max, top_db=40)
        C[0:28]=0
        C[55:]=0
        log_cqt_power = librosa.amplitude_to_db(np.abs(C), ref=np.max, top_db=60)

        cqt_power_max_index = np.argmax(log_cqt_power, axis=0)

        chroma_high = [0] * len(cqt_power_max_index)
        for i in range(len(cqt_power_max_index)):
            index = cqt_power_max_index[i]
            if index == 0:
                chroma_high[i] = 'NoVoice'
            elif index < 40:
                chroma_high[i] = 'l'
            elif index < 50:
                chroma_high[i] = 'm'
            else:
                chroma_high[i] = 'h'

        chroma_pitch = np.argmax(chroma, axis=0)
        log_cqt_power = librosa.amplitude_to_db(np.abs(C), ref=np.max)

        chroma_pitch_high = np.zeros(len(chroma_pitch))
        for i in range(len(chroma_high)):
            if chroma_high[i] == 'NoVoice':
                chroma_pitch_high[i] = np.nan
            elif chroma_high[i] == 'l':
                chroma_pitch_high[i] = chroma_pitch[i]
            elif chroma_high[i] == 'm':
                chroma_pitch_high[i] = chroma_pitch[i] + 12
            elif chroma_high[i] == 'h':
                chroma_pitch_high[i] = chroma_pitch[i] + 24


        init_pitch = chroma_pitch_high[0]
        counter = 0
        pitchcounterlist = []

        for chroma in chroma_pitch_high:
            current_pitch = chroma
            if init_pitch == current_pitch:
                counter+= 1
                init_pitch = chroma
            else:
                pitchcounterlist.append([init_pitch, counter])
                counter = 1
                init_pitch = chroma

        for i in range(len(pitchcounterlist)):
            pitch = pitchcounterlist[i][0]
            counter = pitchcounterlist[i][1]
            if (np.isnan(pitch)==False) & (counter < 5) & (i != 0):
                pitchcounterlist[i][0] = pitchcounterlist[i-1][0]
                pitchcounterlist[i][1] = pitchcounterlist[i-1][1] + counter
                pitchcounterlist[i-1] = 0
            else:
                pass
        new_pitchcounterlist = [i for i in pitchcounterlist if i != 0]

        for i in range(len(new_pitchcounterlist)):
            pitch = new_pitchcounterlist[i][0]
            counter = new_pitchcounterlist[i][1]
            if np.isnan(pitch) or i ==0:
                pass
            elif new_pitchcounterlist[i-1][0] == new_pitchcounterlist[i][0]:
                new_pitchcounterlist[i][0] = new_pitchcounterlist[i-1][0]
                new_pitchcounterlist[i][1] = new_pitchcounterlist[i-1][1] + counter
                new_pitchcounterlist[i-1] = 0
            else:
                pass
        new_pitchcounterlist = [i for i in new_pitchcounterlist if i != 0]

        keys = new_pitchcounterlist

        new_x = voice_dict['NoVoice']
        for key in keys:
            init_pitch = key[0]
            counter = key[1]
            new_x = con_wave(init_pitch, counter, new_x, voice_dict,log_cqt_power,inputsec)
        new_x = con_wave(init_pitch, counter, new_x, voice_dict,log_cqt_power,inputsec)


        raw_data = new_x
        output = struct.pack("h" * len(raw_data), *raw_data)

        wavfile = wave.open(base + filename + ".wav" , "rb" )
        sampling_rate = wavfile.getframerate()
        sample_size = wavfile.getsampwidth()
        wavfile.close()

        folder = base + filename + "_output.wav"
        ww = wave.open(folder, 'w')
        ww.setnchannels(2)
        ww.setsampwidth(sample_size)
        ww.setframerate(sampling_rate)
        ww.writeframes(output)
        ww.close()

    #########################################

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()

    main_window.show()
    sys.exit(app.exec_())
