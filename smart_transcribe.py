
import os
import whisper
import spacy
import openai
import torch
import pyaudio
import numpy as np
import threading

openai.api_key = os.getenv("OPENAI_KEY")
nlp = spacy.load("en_core_sci_lg")
'''
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("loading onto gpu")
    model = whisper.load_model("base.en", mps_device)
else:
    model = whisper.load_model("base.en")
'''
model = whisper.load_model("base.en")

def transcribe(transribe_item):
    result = model.transcribe(transribe_item)['text']
    file1 = open("transcript.txt", "a")  # append mode
    file1.write(result)
    file1.close()

    #print("\nText: " + result)

    doc = nlp(result)
    #print("\nKeywords:" + str(doc.ents))

    for keyword in doc.ents:
        #print(f"Keywork {keyword} is: ")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt= f"In 4 sentences or less what is {keyword}?",
            temperature=0,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0
            )
        #print(response["choices"][0]["text"].replace("\n",""))
        #print("\n")
        response = response["choices"][0]["text"].replace("\n","")
        file2 = open("definitions.txt", "a")  # append mode
        file2.write(f"{keyword}: {response}\n\n")
        file2.close()

CHUNKSIZE = 1024 # fixed chunk size
RATE= 16000
RECORD_SECONDS = 5
# initialize portaudio
p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE, input_device_index=3)
print("starting")
while True: 
    # do this as long as you want fresh samples
    frames = []
    #print("recording")
    for i in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
        data = stream.read(CHUNKSIZE)
        frames.append(data)
    #data = stream.read(CHUNKSIZE)
    #print("done recording")
    numpydata = np.frombuffer(np.array(frames).flatten(), dtype=np.int16).flatten().astype(np.float32) / 32768.0 
    mp = threading.Thread(target=transcribe, args=(numpydata,))
    mp.start()

# close stream
stream.stop_stream()
stream.close()
p.terminate()


