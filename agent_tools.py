from smolagents import tool
import numpy as np
import time
import datetime
import re
from markdownify import markdownify
import requests
from requests.exceptions import RequestException
import litellm
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

@tool
def calculator_tool(expression: str) -> str:
    """
    A simple calculator that performs basic arithmetic operations.

    Args:
        expression: The mathematical expression to evaluate (e.g., '2 + 3 * 4'). 
    """
    # Remove any non-digit or non-operator characters from the expression
    expression = re.sub(r'[^0-9+\-*/().]', '', expression)
    
    try:
        # Evaluate the expression using the built-in eval() function
        result = eval(expression)
        return str(result)
    except (SyntaxError, ZeroDivisionError, NameError, TypeError, OverflowError):
        return "Error: Invalid expression"
    
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    

@tool
def read_excel_file(file_path: str) -> str:
    """Reads an Excel file and returns its content as a string.

    Args:
        file_path: The path to the Excel file.

    Returns:
        The content of the Excel file as a string, or an error message if the file cannot be read.
    """
    try:
        import pandas as pd
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Convert the DataFrame to a string representation
        return df.to_string()

    except Exception as e:
        return f"Error reading the Excel file: {str(e)}"
    
@tool
def read_python_file(file_path: str) -> str:
    """Reads a Python file and returns its content as a string.

    Args:
        file_path: The path to the Python file.

    Returns:
        The content of the Python file as a string, or an error message if the file cannot be read.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading the Python file: {str(e)}"
    

@tool
def read_python_file(file_path: str) -> str:
    """Reads a Python file and returns its content as a string.

    Args:
        file_path: The path to the Python file.

    Returns:
        The content of the Python file as a string, or an error message if the file cannot be read.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading the Python file: {str(e)}"

import base64

def convert_image_to_base64(image_path: str) -> str:
    """
    Converts an image to a Base64 string.

    Args:
        image_path: The path to the image file.

    Returns:
        A Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            # Read the image file as binary data
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        return f"Error converting image to Base64: {str(e)}"



@tool
def describe_image(image_path: str) -> str:
    """
    Return the description of a given image.

    Args:
        image_path: the input path of the image to describe.
    """

    encoded_data = convert_image_to_base64(image_path)

    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe the image in detail."
            },
            
            # {
            #      "type": "image_url",
            #      "image_url": {"url": image_path}
            #  }
            {
                    "type": "file",
                    "file": {
                        "file_data": f"data:image/png;base64,{encoded_data}",  # 👈 SET MIME_TYPE + DATA
                    }
                },
        ]
    }
]

    # Make the API call to Gemini model
    response = litellm.completion(
        model="gemini/gemini-2.0-flash-lite",
        messages=messages,
    )

    # Extract the response content
    content = response.get('choices', [{}])[0].get('message', {}).get('content')

    return content

@tool
def describe_audio(audio_path: str) -> str:
    """
    Return the transcription of a given audio file.

    Args:
        audio_path: the input path of the audio to transcribe.
    """

    audio_bytes = Path(audio_path).read_bytes()
    encoded_data = base64.b64encode(audio_bytes).decode("utf-8")

    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please transcribe the content of this audio."},
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:audio/mp3;base64,{}".format(encoded_data), # 👈 SET MIME_TYPE + DATA
                    }
                },
            ],
        }
    ]

    # Make the API call to Gemini model
    response = litellm.completion(
        model="gemini/gemini-2.0-flash-lite",
        messages=messages,
    )

    # Extract the response content
    content = response.get('choices', [{}])[0].get('message', {}).get('content')

    return content
    

if __name__ == "__main__":
    # example usage of image description
    image_path = "./cca530fc-4052-43b2-b130-b30968d8aa44.png"
    description = describe_image(image_path)
    print(description)