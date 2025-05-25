import requests

def download_file(record):
    DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

    task_id = record['task_id']
    url = DEFAULT_API_URL+f'/files/{task_id}'

    filename = record['file_name']

    if filename.endswith('.mp3'):
        # Local path where the audio will be saved        
        # Send a GET request to fetch the audio file
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Open a local file in binary write mode
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            print(f"Audio file downloaded successfully and saved as {filename}")
        else:
            print(f"Failed to download audio file. Status code: {response.status_code}")

    elif filename.endswith('.png') or filename.endswith('.py') or filename.endswith('.xlsx'):
        # Send a GET request to fetch the image file
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Open a local file in binary write mode
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            print(f"Image file downloaded successfully and saved as {filename}")
        else:
            print(f"Failed to download image file. Status code: {response.status_code}")