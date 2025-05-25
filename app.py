import os
import gradio as gr
import requests
import inspect
import pandas as pd
from smolagents import CodeAgent, LiteLLMModel, Tool, DuckDuckGoSearchTool
import litellm
from agent_tools import calculator_tool, visit_webpage, read_excel_file, read_python_file, describe_audio, describe_image
import numpy as np
import time
import datetime
from utils import download_file
from dotenv import load_dotenv
import json

load_dotenv()

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):

        print("BasicAgent initialized.")
        # Initialize the Hugging Face model
        self.model = llm_model = LiteLLMModel(
                model_id="gemini/gemini-2.0-flash", # you can see other model names here: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models. It is important to prefix the name with "gemini/"
                api_key=os.getenv("GEMINI_API_KEY"),
                max_tokens=8192
            )
        search_tool = DuckDuckGoSearchTool()

        self.web_agent = CodeAgent(
            tools=[search_tool, visit_webpage],
            model=self.model,
            max_steps=5,
            name="web_search_agent",
            description="Runs web searches for you.",
        )

    def check_final_answer(self, final_answer, agent_memory):
        prompt = f"""Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. \ 
                    Report your thoughts, and finish your answer with the following template:
                    FINAL ANSWER: [YOUR FINAL ANSWER]. 
                    YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
                    If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign \ 
                    unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), \
                    and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list,\
                    apply the above rules depending of whether the element to be put in the list is a number or a string.
                    """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        #model = InferenceClientModel()

        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            #model="gemini/gemini-2.0-flash-lite",
            messages=messages,
        )

        # Extract the response content
        content = response.get('choices', [{}])[0].get('message', {}).get('content')

        return content
    
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        self.manager_agent = CodeAgent(
            tools=[calculator_tool],
            managed_agents=[self.web_agent, describe_audio, describe_image, read_excel_file, read_python_file],
            model=self.model,
            additional_authorized_imports=["pandas", "re", "requests", "json", "numpy", "bs4", "datetime", "os", "io", "csv"],
            max_steps=10,
            verbosity_level=2,
            planning_interval=5,
            add_base_tools=True,
            final_answer_checks=[self.check_final_answer]
        )
        #fixed_answer = "This is a default answer."
        #print(f"Agent returning fixed answer: {fixed_answer}")
        answer = self.manager_agent.run(question)
        return answer

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if len(item.get("file_name")):
            download_file(item)
            question_text += f" The file path is: {item.get('file_name')}"
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})

        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})
        
        time.sleep(100)

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)
    
    # 6. Save answers to json
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"answers_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(answers_payload, f)
        print(f"Answers saved to {filename}")
    except Exception as e:
        print(f"Error saving answers to file: {e}")

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df    




# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# My Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        2.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)