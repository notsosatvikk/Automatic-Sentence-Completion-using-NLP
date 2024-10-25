import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import speech_recognition as sr
import time

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add background color and custom styles
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
    }
    body {
        background-image: url('https://www.example.com/background.jpg');
        background-size: cover;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: #4b8bbe;
        font-family: 'Arial', sans-serif;
    }
    .subtitle {
        font-size: 30px;
        text-align: center;
        font-family: 'Verdana', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #4b8bbe;
        border-radius: 10px;
        width: 200px;
        height: 50px;
        font-size: 20px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .generated-box {
        border: 2px solid #4b8bbe;
        border-radius: 10px;
        padding: 10px;
        color: #333;
    }
    .generated-text {
        color: #4b8bbe;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App title and subtitle
st.markdown('<h1 class="title">Let Me Do It For You !</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Next Word Prediction App</h3>', unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.title("Settings")
input_method = st.sidebar.radio("Choose input method:", ("Type", "Speak"))
num_words = st.sidebar.number_input("Number of words to predict", min_value=1, max_value=50, value=5)

# Initialize input_text as an empty string
input_text = ""

# Function to recognize speech and return it as text
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
    try:
        st.write("Processing speech...")
        recognized_text = recognizer.recognize_google(audio)
        return recognized_text
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that. Please try again."
    except sr.RequestError:
        return "Request to Google Speech Recognition failed."

# Layout using columns for input method
col1, col2 = st.columns(2)

# If the user chooses to type, display a text input box
if input_method == "Type":
    with col1:
        input_text = st.text_input("Type some text:")

# If the user chooses to speak, display the button in the center
elif input_method == "Speak":
    st.write("")
    if st.button("Speak Now"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Simulate loading time
            progress.progress(i + 1)
        input_text = recognize_speech()
        st.write(f"You said: **{input_text}**")

# Function to predict logical next words with top-k and temperature control
def predict_next_words(input_text, num_words, top_k=50, temperature=0.7, num_return_sequences=4):
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate the next num_words tokens with top-k sampling and temperature control
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + num_words,
            num_return_sequences=num_return_sequences,  # Return 4 different sequences
            do_sample=True,              # Enable sampling for more natural results
            top_k=top_k,                 # Top-k sampling to limit token choices
            temperature=temperature,     # Control randomness
            eos_token_id=tokenizer.eos_token_id,  # Stop at the end of a sentence
            pad_token_id=tokenizer.eos_token_id   # Ensure sentences are padded logically
        )
    
    # Decode the generated sequences and return the result
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True)[len(input_text):].strip() for seq in output]
    return generated_texts

# Only run the prediction if there is input text
if input_text:
    st.write("Generating four logical continuations...")
    # Generate the text
    generated_texts = predict_next_words(input_text, num_words)
    
    # Display the four options
    for i, option in enumerate(generated_texts, 1):
        st.markdown(
            f"""
            <div class="generated-box">
                <p>Option {i}:</p>
                <h2 class="generated-text">{option}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
