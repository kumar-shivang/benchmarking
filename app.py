import streamlit as st
import json
import base64
import os
from llm import LLM
from dotenv import load_dotenv
from schemas import EVALUATION_SCHEMA
from logger import setup_logger, log_exception, log_api_call

# Load environment variables
load_dotenv(override=True)

# Setup logging for Streamlit app
logger = setup_logger("streamlit_app", "logs")
logger.info("Streamlit app started")

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
        logger.info(
            f"Initialized models - Student: {slave_model_name}, Evaluator: {master_model_name}"
        )
    else:
        st.sidebar.warning("Please provide an API Key.")
        logger.warning("No API key provided")
        slave_model = None
        master_model = None
except Exception as e:
    st.sidebar.error(f"Error initializing LLM: {e}")
    log_exception(logger, e, "Initializing LLM models in Streamlit")
    slave_model = None
    master_model = None


def generate_critique(question, student_answer, original_answer):
    try:
        logger.debug("Generating critique for question")
        with open("evaluation_prompt.md", "r") as f:
            prompt_template = f.read()

        # Ensure student_answer is a dict for formatting
        # if isinstance(student_answer, str):
        #     try:
        #         student_answer = json.loads(student_answer)
        #     except json.JSONDecodeError:
        #         student_answer = {
        #             "step_by_step_reasoning": "N/A",
        #             "final_answer": student_answer,
        #         }

        prompt = prompt_template.format(
            question=question, answer=student_answer, original_answer=original_answer
        )
    except Exception as e:
        st.error(f"Error reading evaluation_prompt.md: {e}")
        log_exception(logger, e, "Reading evaluation prompt")
        # Fallback to a basic prompt if file reading fails
        prompt = f"Evaluate this answer. Question: {question}, Student: {student_answer}, Original: {original_answer}"

    try:
        response, usage = master_model.generate_response(
            [
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt},
            ],
            response_format=EVALUATION_SCHEMA,
        )
        log_api_call(
            logger,
            master_model.model_name,
            "Generate critique",
            success=True,
            details=f"Cost: ${usage.get('cost', 0):.4f}",
        )
        return response
    except Exception as e:
        log_exception(logger, e, "Generating critique")
        raise


# Initialize session state for storing results
if "results" not in st.session_state:
    st.session_state.results = {}

# File Uploader
uploaded_file = st.file_uploader("Upload JSON File", type=["json"])

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)
        logger.info(
            f"Loaded JSON file: {uploaded_file.name} with {len(data) if isinstance(data, list) else 1} items"
        )

        if not isinstance(data, list):
            st.error("Invalid JSON format. Expected a list of objects.")
            logger.error(f"Invalid JSON format in {uploaded_file.name}")
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
                                    try:
                                        logger.info(
                                            f"Generating answer for question {i+1}"
                                        )
                                        student_answer, usage = (
                                            slave_model.generate_response(
                                                [
                                                    {
                                                        "role": "system",
                                                        "content": "Answer the question.",
                                                    },
                                                    {
                                                        "role": "user",
                                                        "content": item.get("question"),
                                                    },
                                                ],
                                                # response_format=STUDENT_ANSWER_SCHEMA,
                                            )
                                        )
                                        # try:
                                        #     student_answer = json.loads(student_answer)
                                        # except Exception:
                                        #     pass
                                        st.session_state.results[question_id][
                                            "student_answer"
                                        ] = student_answer
                                        st.session_state.results[question_id][
                                            "critique"
                                        ] = None  # Reset critique if answer changes
                                        log_api_call(
                                            logger,
                                            slave_model.model_name,
                                            f"Generate answer {i+1}",
                                            success=True,
                                            details=f"Cost: ${usage.get('cost', 0):.4f}",
                                        )
                                    except Exception as e:
                                        st.error(f"Error generating answer: {e}")
                                        log_exception(
                                            logger,
                                            e,
                                            f"Generating answer for question {i+1}",
                                        )
                            else:
                                st.error(
                                    "Slave LLM not initialized. Please check API Key."
                                )
                                logger.error(
                                    "Attempted to generate answer without initialized slave model"
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
                                    try:
                                        logger.info(
                                            f"Evaluating answer for question {i+1}"
                                        )
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
                                        logger.info(
                                            f"Successfully evaluated question {i+1}"
                                        )
                                    except Exception as e:
                                        st.error(f"Error generating critique: {e}")
                                        log_exception(
                                            logger, e, f"Evaluating question {i+1}"
                                        )
                            else:
                                st.error(
                                    "Master LLM not initialized. Please check API Key."
                                )
                                logger.error(
                                    "Attempted to evaluate without initialized master model"
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
