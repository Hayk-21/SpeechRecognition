from keras.models import load_model
import numpy as np
import sounddevice as sd
import time
import multiprocessing as mp


model=load_model('barevModel.h5')
classes = ['background_noise', 'barev']
samplerate = 8000

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

def func_1():
    while True:
        recording = sd.rec(int(1*samplerate),samplerate = samplerate , channels = 1, dtype='float32').ravel()
        sd.wait()
        # recording = librosa.resample(recording, orig_sr=16000, target_sr=8000)
# recording = np.array(recording).reshape(8000,1)
        if predict(recording) == "barev":
            print("barev")


def func_2():
    while True:
        time.sleep(0.3)
        recording = sd.rec(int(1*samplerate),samplerate = samplerate , channels = 1, dtype='float32').ravel()
        sd.wait()
        # recording = librosa.resample(recording, orig_sr=samplerate, target_sr=8000)
    # recording = np.array(recording).reshape(8000,1)
        if predict(recording) == "barev":
            print("barev")

def func_3():
    while True:
        time.sleep(0.6)
        recording = sd.rec(int(1*samplerate),samplerate = samplerate , channels = 1, dtype='float32').ravel()
        sd.wait()
        # recording = librosa.resample(recording, orig_sr=16000, target_sr=8000)
    # recording = np.array(recording).reshape(8000,1)
        if predict(recording) == "barev":
            print("barev")

if __name__ == '__main__':
    proc_1 = mp.Process(target=func_1)
    proc_2 = mp.Process(target=func_2)
    proc_3 = mp.Process(target=func_3)
    
    proc_1.start()
    proc_2.start()
    proc_3.start()