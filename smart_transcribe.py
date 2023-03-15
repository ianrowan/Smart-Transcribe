
import os
import whisper
import spacy
import openai
import pyaudio
import numpy as np
import threading
import multiprocessing

openai.api_key = os.getenv("OPENAI_KEY")
nlp = spacy.load("en_core_sci_lg")
#nlp.select_pipes(enable="ner")
#print(nlp.get_pipe("ner").cfg)
#nlp.get_pipe("ner").cfg["beam_density"] = 0.00005

model = whisper.load_model("base.en")
curr_end_token = 0
frame_start_token = 0

def gpt_keyword_query(keywords, turbo):
    filer = open("definitions.txt", "r")
    curr_keywords = filer.read()
    filew = open("definitions.txt", "a")  # append mode
    #curr_keywords = "None"
    for keyword in keywords:
        if(f"{keyword}:" not in curr_keywords):
            #print(f"Keywork {keyword} is: ")
            if turbo:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                    {"role": "system", "content": "You are a definition summarization bot"},
                    {"role": "user", "content": f"In 4 sentences or less explain {keyword}"}
                    ]
                )
                response = response["choices"][0]["message"]["content"].replace("\n","")
                #print(response["choices"][0]["message"]["content"])
                #print("\n")
            else:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt= f"In 4 sentences or less what is {keyword}?",
                    temperature=0,
                    max_tokens=60,
                    top_p=1,
                    frequency_penalty=0.5,
                    presence_penalty=0
                    )
                print(response["choices"][0]["text"].replace("\n",""))
                print("\n")
                response = response["choices"][0]["text"].replace("\n","")
            filew.write(f"{keyword}: {response}\n\n")
        else:
            print(f"skipping {keyword}")
    filew.close()
    filer.close()

def transcribe(transribe_item, model, new_split):
    #1. Transcribe all text and overwrite file
    result = model.transcribe(transribe_item)['text']
    global frame_start_token
    f2 = open("transcript.txt", "r")
    content = f2.read()
    file1 = open("transcript.txt", "w")
    if frame_start_token > 0 or new_split:
        if new_split:
            print(len(content))
            frame_start_token = len(content)
        #print("result" + result)
        #print("content" + content[:frame_start_token])
        new_content = content[:frame_start_token] + result
        f2.close()
        file1.write(new_content)  
    else:
        file1.write(result)
    file1.close()
    #print("\nText: " + result)
    global curr_end_token
    #2. Extract all new keywords(ending token of prev), set new previous token
    doc = nlp(result[curr_end_token:])
    #print("\nKeywords:" + str(doc.ents))
    curr_end_token = len(result)

    #3. for each keyword, check if exists already and look up def if not 
    print(doc.ents)
    t1 = threading.Thread(target=gpt_keyword_query, args=(doc.ents, True,))
    t1.start()

def main():
    CHUNKSIZE = 1024 # fixed chunk size
    RATE= 16000
    RECORD_SECONDS = 5
    # initialize portaudio
    p = pyaudio.PyAudio()

    input_i = 0
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if("Black" in dev["name"]):
            print(dev)
            input_i = dev["index"]
            break
        #print(p.get_device_info_by_index(i))

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE, input_device_index=input_i)

    print("starting")
    frames = []
    sec_count = 0
    try:
        while True: 
            #print("recording")
            reset_frame = sec_count != 0 and (sec_count % 30 == 0)
            if reset_frame:
                frames = []
            for i in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
                data = stream.read(CHUNKSIZE)
                frames.append(data)
            sec_count += 5
            #print("done recording")
            
            numpydata = np.frombuffer(np.array(frames).flatten(), dtype=np.int16).flatten().astype(np.float32) / 32768.0 
            #mp = multiprocessing.Process(target=transcribe, args=(numpydata, model, reset_frame))
            mp = threading.Thread(target=transcribe, args=(numpydata, model, reset_frame))
            mp.start()

    except KeyboardInterrupt:
        # close stream
        stream.stop_stream()
        stream.close()
        p.terminate()

main()