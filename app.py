import streamlit as st
import json
import base64
import os
from llm import LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

st.set_page_config(page_title="LLM Evaluation Benchmarking", layout="wide")

st.title("LLM Evaluation Benchmarking")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", "")
)

if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key


# Get model list
@st.cache_data(show_spinner="Fetching models...")
def get_available_models(api_key):
    try:
        if not api_key:
            return []
        # Create a temporary LLM instance just to get models
        temp_llm = LLM()
        return temp_llm.get_models()
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {e}")
        return []


model_list = get_available_models(api_key)
if model_list:
    # Try to find a sensible default
    default_model = "deepseek/deepseek-v3.2"
    if default_model not in model_list:
        default_model = "deepseek/deepseek-v3"

    default_index = (
        model_list.index(default_model) if default_model in model_list else 0
    )

    st.sidebar.subheader("Slave Model (Answering)")
    slave_model_name = st.sidebar.selectbox(
        "Select Slave Model", options=model_list, index=default_index, key="slave_model"
    )

    st.sidebar.subheader("Master Model (Evaluating)")
    master_model_name = st.sidebar.selectbox(
        "Select Master Model",
        options=model_list,
        index=default_index,
        key="master_model",
    )
else:
    slave_model_name = st.sidebar.text_input(
        "Slave Model Name", value="deepseek/deepseek-v3.2", key="slave_model_text"
    )
    master_model_name = st.sidebar.text_input(
        "Master Model Name", value="deepseek/deepseek-v3.2", key="master_model_text"
    )

# Initialize LLMs
try:
    if api_key:
        slave_model = LLM(slave_model_name)
        master_model = LLM(master_model_name)
    else:
        st.sidebar.warning("Please provide an API Key.")
        slave_model = None
        master_model = None
except Exception as e:
    st.sidebar.error(f"Error initializing LLM: {e}")
    slave_model = None
    master_model = None


def generate_critique(question, student_answer, original_answer):
    try:
        with open("evaluation_prompt.md", "r") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(
            question=question, answer=student_answer, original_answer=original_answer
        )
    except Exception as e:
        st.error(f"Error reading evaluation_prompt.md: {e}")
        # Fallback to a basic prompt if file reading fails
        prompt = f"Evaluate this answer. Question: {question}, Student: {student_answer}, Original: {original_answer}"

    response = master_model.generate_response(
        [
            {"role": "system", "content": "You are an expert evaluator."},
            {"role": "user", "content": prompt},
        ]
    )
    return response


# Initialize session state for storing results
if "results" not in st.session_state:
    st.session_state.results = {}

# File Uploader
uploaded_file = st.file_uploader("Upload JSON File", type=["json"])

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)

        if not isinstance(data, list):
            st.error("Invalid JSON format. Expected a list of objects.")
        else:
            st.success(f"Loaded {len(data)} items.")

            for i, item in enumerate(data):
                question_id = f"q_{i}_{os.path.basename(uploaded_file.name)}"
                if question_id not in st.session_state.results:
                    st.session_state.results[question_id] = {
                        "student_answer": None,
                        "critique": None,
                    }

                with st.expander(
                    f"Question {i+1}: {item.get('question')[:50]}...", expanded=False
                ):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("Question")
                        st.markdown(item.get("question", "No question provided."))

                        if "image" in item and item["image"]:
                            try:
                                # Check if image is base64 string
                                image_data = base64.b64decode(item["image"])
                                st.image(image_data, caption="Question Image")
                            except Exception as e:
                                st.error(f"Error decoding image: {e}")

                    with col2:
                        st.subheader("Original Solution")
                        st.markdown(item.get("solution", "No solution provided."))

                    st.divider()

                    gen_col, eval_col = st.columns(2)

                    with gen_col:
                        if st.button(f"Generate Answer {i+1}", key=f"gen_{i}"):
                            if slave_model:
                                with st.spinner("Generating Answer..."):
                                    student_answer = slave_model.generate_response(
                                        [
                                            {
                                                "role": "system",
                                                "content": "Answer the question.",
                                            },
                                            {
                                                "role": "user",
                                                "content": item.get("question"),
                                            },
                                        ]
                                    )
                                    st.session_state.results[question_id][
                                        "student_answer"
                                    ] = student_answer
                                    st.session_state.results[question_id][
                                        "critique"
                                    ] = None  # Reset critique if answer changes
                            else:
                                st.error(
                                    "Slave LLM not initialized. Please check API Key."
                                )

                    with eval_col:
                        # Only enable evaluate if an answer exists
                        can_evaluate = (
                            st.session_state.results[question_id]["student_answer"]
                            is not None
                        )
                        if st.button(
                            f"Evaluate Answer {i+1}",
                            key=f"eval_{i}",
                            disabled=not can_evaluate,
                        ):
                            if master_model:
                                with st.spinner("Generating Critique..."):
                                    student_answer = st.session_state.results[
                                        question_id
                                    ]["student_answer"]
                                    critique = generate_critique(
                                        item.get("question"),
                                        student_answer,
                                        item.get("solution"),
                                    )
                                    st.session_state.results[question_id][
                                        "critique"
                                    ] = critique
                            else:
                                st.error(
                                    "Master LLM not initialized. Please check API Key."
                                )

                    # Display results if they exist
                    if st.session_state.results[question_id]["student_answer"]:
                        st.markdown("---")
                        st.markdown("### Student Answer")
                        st.markdown(
                            st.session_state.results[question_id]["student_answer"]
                        )

                    if st.session_state.results[question_id]["critique"]:
                        st.markdown("### Critique")
                        st.markdown(st.session_state.results[question_id]["critique"])

    except json.JSONDecodeError:
        st.error("Invalid JSON file.")
