import json
import os
import logging
import pypandoc
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

import streamlit as st

import cv2
import pytesseract
import cv2
from PIL import Image
import os
import numpy as np
import whisper
import datetime
from bson import json_util

from pydub import AudioSegment
import pandas as pd

# change the link to the location of tesseract locally
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("ContextLLM")

from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    PERSIST_DIRECTORY,
    MAX_NEW_TOKENS,
    MODELS_PATH,
)
    
def load_model(device_type, temp, topp, model_id, model_basename=None, LOGGING=logging):
    
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    generation_config = GenerationConfig.from_pretrained(model_id)

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=temp,
        top_p=topp,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(EMBEDDING_MODEL_NAME, MODEL_ID, MODEL_BASENAME, device_type, temp, topp, use_history, promptTemplate_type="llama"):
    
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, temp, topp, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        callbacks=callback_manager,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    return qa

def save_audio_file(audio_bytes, file_extension):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name

def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

def to_mp3(audio_file, output_audio_file, upload_path, download_path):
    ## Converting Different Audio Formats To MP3 ##
    if audio_file.name.split('.')[-1].lower()=="wav":
        audio_data = AudioSegment.from_wav(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="mp3":
        audio_data = AudioSegment.from_mp3(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="ogg":
        audio_data = AudioSegment.from_ogg(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="wma":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"wma")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="aac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"aac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="flac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"flac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="flv":
        audio_data = AudioSegment.from_flv(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="mp4":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"mp4")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")
    return output_audio_file

def save_transcript(transcript_data, txt_file):
    with open(os.path.join("SOURCE_DOCUMENTS/", txt_file),"w") as f:
        f.write(transcript_data)


if __name__ == "__main__":
    def delete():
        for i in range(0, len(files)):
            if(st.session_state.present == []):
                # print(st.session_state.present)
                # print(files[i].name)
                if os.path.exists("SOURCE_DOCUMENTS/"+files[i].name.split('.')[0]+".txt"):
                    os.remove("SOURCE_DOCUMENTS/"+files[i].name.split('.')[0]+".txt")
                else:
                    os.remove("SOURCE_DOCUMENTS/"+files[i].name)
            
    with st.sidebar:
        st.subheader("Upload Documents to Provide Context")
        files = st.file_uploader("Upload Documents", ['png', 'jpg','jpeg','pdf',
                                            'txt', 'csv', 'xlsx', 'xls', 
                                            'doc', 'docx', 'mp3', 'mp4'], True, key="present", on_change=delete)
            
        for i in range(0, len(files)):
            # print(files[i].type)
            # print(st.session_state.present)
            if("image" in files[i].type):
                
                image = Image.open(files[i])
                image_array = np.array(image)
                image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(image)
                with open("SOURCE_DOCUMENTS/"+files[i].name.split('.')[0]+".txt", 'w') as f:
                    f.write(text)
            
            elif("video" in files[i].type or "audio" in files[i].type):
                with st.spinner(f"Processing Audio"):
                    audio_bytes = files[i].read()
                    with open(os.path.join("inputs/",files[i].name),"wb") as f:
                        f.write((files[i]).getbuffer())
                    
                    output_audio_file = files[i].name.split('.')[0] + '.mp3'
                    output_audio_file = to_mp3(files[i], output_audio_file, "inputs/", "inputs/")
                    audio_file = open(os.path.join("inputs/",output_audio_file), 'rb')
                    audio_bytes = audio_file.read()
                    transcript = process_audio(str(os.path.abspath(os.path.join("inputs/",output_audio_file))), "base.en")
                    output_txt_file = str(output_audio_file.split('.')[0]+".txt")
                    save_transcript(transcript, output_txt_file)
            
            elif(files[i].name.split('.')[1] == 'csv'):
                df = pd.read_csv(files[i])
                df.to_csv("SOURCE_DOCUMENTS/"+files[i].name, sep='\t')
            
            elif(files[i].name.split('.')[1] == 'txt'):
                raw_text = str(files[i].read(), "utf-8")
                with open("SOURCE_DOCUMENTS/"+files[i].name, "w") as f:
                    f.write(raw_text)
                    
            elif(files[i].name.split('.')[1] == 'doc' or files[i].name.split('.')[1] == 'docx'):
                pypandoc.convert_file(files[i].name, 'plain', outputfile="SOURCE_DOCUMENTS/"+files[i].name.split(".")[0]+".txt")
            
            elif(files[i].name.split('.')[1] == 'pdf'):
                with open("SOURCE_DOCUMENTS/"+files[i].name, "wb") as w:
                    w.write(files[i].getvalue())
                
            elif(files[i].name.split('.')[1] == 'xls' or files[i].name.split('.')[1] == 'xlsx'):
                df = pd.read_excel(files[i])
                df.to_excel("SOURCE_DOCUMENTS/"+files[i].name)
        
        st.subheader("Running Parameters")
        selected_model = st.sidebar.selectbox('Choose a Model', ['Llama-2-7b-Chat-GGUF', 'Llama-2-13b-Chat-GGUF', 'vicuna-13B-v1.5-16K-GGUF'], key='selected_model')
        
        if(selected_model == "Llama-2-7b-Chat-GGUF"):
            MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
            MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
        elif(selected_model == "Llama-2-13b-Chat-GGUF"):
            MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
            MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"
        elif(selected_model=="Wizard-vicuna-13B-GGML"):
            MODEL_ID = "TheBloke/vicuna-13B-v1.5-16K-GGUF"
            MODEL_BASENAME = "vicuna-13b-v1.5-16k.Q4_K_M.gguf"
            
        selected_embedding = st.sidebar.selectbox('Choose an Embeddings Model', ['hkunlp/instructor-large', 'all-MiniLM-L6-v2', 'intfloat/e5-base-v2'], key='selected_embedding')
        if(selected_embedding == "hkunlp/instructor-large"):
            EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
        elif(selected_embedding == "all-MiniLM-L6-v2"):
            EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        elif(selected_embedding == "intfloat/e5-base-v2"):
            EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
            
        running = st.sidebar.selectbox('Do you want to run on: ', ['cpu', 'cuda', 'mps'], key='running_hardware')
        temp = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        topp = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        def generate():
            import os, shutil
            folder = 'DB/'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
                    
            os.system('python ingest.py --device_type '+running+" --embedding_model_name "+EMBEDDING_MODEL_NAME)
            
        def download_chats():
            with open("model_chats.txt", "w") as f:
                f.write(str(st.session_state.messages))
        
        def reset_chats():
            print(st.session_state)
            st.session_state.messages = []
            print(st.session_state)
            
        st.button("Generate Embeddings", on_click=generate)
        st.button("Download Chats", on_click=download_chats)
        st.button("Reset Chats", on_click=reset_chats)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    if prompt := st.chat_input("Input?"):
        print(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            finalPrompt = str(prompt)
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if not os.path.exists(MODELS_PATH):
                os.mkdir(MODELS_PATH)
            if(MODEL_BASENAME == "vicuna-13b-v1.5-16k.Q4_K_M.gguf"):
                qa = retrieval_qa_pipline(EMBEDDING_MODEL_NAME, MODEL_ID, MODEL_BASENAME, running, temp, topp, True, promptTemplate_type="vicuna")
            else:
                qa = retrieval_qa_pipline(EMBEDDING_MODEL_NAME, MODEL_ID, MODEL_BASENAME, running, temp, topp,True, promptTemplate_type="llama")
            # Exit Condition
            if prompt == "exit":
                message_placeholder.markdown("I hope I was able to get you the answers you were looking for")
                exit()
                
            res = qa(prompt)
            answer, docs = res["result"], res["source_documents"]
            message_placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
            
        # Display sources -> have to check
                    # print("----------------------------------SOURCE DOCUMENTS---------------------------")
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)
        # print("----------------------------------SOURCE DOCUMENTS---------------------------")

        # Device to run on
            
            
    