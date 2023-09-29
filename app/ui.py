import requests
import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

BACKEND_URL = "http://localhost:8000/predict"
GREETING_MESSAGE = (
    """Hello there, Plant Pal user! 🌿🤖\n\n"""
    """I'm PlantPal, your trusty AI-based house plant doctor, and I'm here to help you keep your leafy friends healthy and thriving. Whether you're a seasoned plant parent or just starting out on your green journey, I've got you covered.\n\n"""
    """Feel free to ask me anything about your plants - from watering schedules to sunlight requirements, disease identification, or even general plant care tips. I'm here to make sure your indoor jungle stays lush and vibrant.\n\n"""
    """Let's embark on this green adventure together! Ask away, and let's help your plants flourish. 🪴🌱💚\n\n"""
)

st.set_page_config(page_title="PlantPal - An LLM-powered PlantAdvisor")

# Sidebar contents
with st.sidebar:
    st.title("🤗🌳💬 PlantPal App")
    add_vertical_space(5)
    st.write("Made with ❤️ by **PlantPal Team**")

# Generate empty lists for generated and past.
## generated stores AI generated responses
if "generated" not in st.session_state:
    st.session_state["generated"] = []

## past stores User's questions
if "past" not in st.session_state:
    st.session_state["past"] = []

## request - response history
if "request" not in st.session_state:
    st.session_state["request"] = []

# Layout of input/response containers
input_container = st.container()
colored_header(label="", description="", color_name="blue-30")
greeting_container = st.container()
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


## Applying the user input box
with input_container:
    user_input = get_text()

with greeting_container:
    st.write(GREETING_MESSAGE)


# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    data = {"text": prompt, "command": "intent", "info": {}, "prev_intent": "NONE"}

    last_req = (
        st.session_state["request"][-1]
        if len(st.session_state["request"]) > 0
        else None
    )
    if last_req:
        if last_req["command"].startswith("CONT"):
            data["command"] = last_req["command"]
            data["prev_intent"] = last_req["prev_intent"]
            data["info"] = last_req["info"]
    print("Request data", data)
    response = requests.post(BACKEND_URL, json=data)

    prediction = response.json()
    print("Response", prediction)
    response = prediction["prediction"]
    st.session_state["request"].append(prediction)
    return response


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
