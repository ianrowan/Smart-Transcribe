
import os
import whisper
import spacy
import openai
openai.api_key = os.getenv("OPENAI_KEY")
nlp = spacy.load("en_core_sci_lg")
model = whisper.load_model("base.en")

result = model.transcribe("test_file6.m4a")['text']

print("\nText: " + result)

doc = nlp(result)
print("\nKeywords:" + str(doc.ents))

for keyword in doc.ents:
    print(f"Keywork {keyword} is: ")
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
