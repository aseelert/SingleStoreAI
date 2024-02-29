#!/home/watsonx/miniconda3/bin/python3
import os
import json
import base64
import platform
import argparse
import pymysql
import openai
import pandas as pd
from getpass import getpass
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
from tabulate import tabulate
from textblob import TextBlob
import nltk
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from tqdm import tqdm 
import requests

nltk.download('punkt')

comment_limit = "30"

ibmaillm = "FLAN_UL2"
model_id = getattr(ModelTypes, ibmaillm)


# Argument Parsing
parser = argparse.ArgumentParser(description='Sentiment Analysis with Secure Credentials')
parser.add_argument('--master-password', type=str, help='Master password for the credential store')
parser.add_argument('-aitype', choices=['openai', 'ibmai', 'both'], required=True, help='AI type for sentiment analysis')
args = parser.parse_args()

SALT_FILE = 'salt.key'
CREDENTIALS_FILE = 'credentials.enc'

def download_file_if_not_exists(url, filename):
    # Check if the file already exists
    if not os.path.exists(filename):
        print(f"{filename} not found. Downloading from {url}")
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            # Raise an HTTPError if the response was unsuccessful
            response.raise_for_status()
            
            # Write the content of the response to a file
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"{filename} has been downloaded.")
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error occurred: {err}")
        except requests.exceptions.RequestException as err:
            print(f"Error downloading the file: {err}")
    else:
        print(f"{filename} already exists in the current directory.")

# URL of the file you want to download
url = "https://portal.singlestore.com/static/ca/singlestore_bundle.pem"
# The filename to save the downloaded file as
filename = "singlestore_bundle.pem"

# Call the function
download_file_if_not_exists(url, filename)

def analyze_sentiment_with_textblob(comment):
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    # Determine sentiment based on polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
    
# Utility Functions
def clear_screen():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

def get_salt():
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, 'rb') as file:
            return file.read()
    else:
        salt = os.urandom(16)
        with open(SALT_FILE, 'wb') as file:
            file.write(salt)
        return salt

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def get_credentials(master_key, service_type):
    try:
        with open(CREDENTIALS_FILE, 'rb') as file:
            encrypted_data = file.read()
        f = Fernet(master_key)
        decrypted_data = f.decrypt(encrypted_data)
        credentials = json.loads(decrypted_data.decode())
        return credentials.get(service_type, {})
    except Exception as e:
        print(f"Error retrieving credentials for {service_type}: {e}")
        return {}

# Credential Retrieval and Connection Setup
if not os.path.exists(CREDENTIALS_FILE) or not os.path.exists(SALT_FILE):
    print("Credential files not found.")
    exit()

master_password = args.master_password or getpass("Enter master password: ")
salt = get_salt()
master_key = derive_key(master_password, salt)

# Initialize credentials dictionary
credentials = {}

# Initialize a list to hold credential status
credentials_status = []

# Load credentials based on the ai_type argument and update credentials_status
if args.aitype in ['openai', 'both']:
    openai_credentials = get_credentials(master_key, 'openai')
    if not openai_credentials:
        credentials_status.append(["OpenAI", "Not Provided", "\033[91m✗\033[0m"])  # Red cross for not provided
    else:
        credentials_status.append(["OpenAI", "Provided", "\033[92m✓\033[0m"])  # Green check mark for provided
        openai_key = openai_credentials.get('api_key')

if args.aitype in ['ibmai', 'both']:
    ibmai_credentials = get_credentials(master_key, 'ibmai')
    if not ibmai_credentials:
        credentials_status.append(["watsonx.ai", "Not Provided", "\033[91m✗\033[0m"])
    else:
        credentials_status.append(["watsonx.ai", "Provided", "\033[92m✓\033[0m"])
        watsonx_api_url = "https://us-south.ml.cloud.ibm.com"
        ibmai_key = ibmai_credentials.get('watsonxai_ibmai_iamkey')
        ibmai_project = ibmai_credentials.get('watsonxai_project_id')
        credentials_status.append(["watsonx.ai", "Provided", "\033[92m✓\033[0m"])
        credentials_status.append(["watsonx.ai", "Provided", "\033[92m✓\033[0m"])
        credentials_status.append(["watsonx.ai", "Provided", "\033[92m✓\033[0m"])

# Print the credentials status in a table format
print(tabulate(credentials_status, headers=["Service", "Status", "Loaded"], tablefmt="grid"))

# Load SingleStore credentials if needed for your application, independent of AI type

credentials['singlestore'] = get_credentials(master_key, 'singlestore')


if args.aitype in ['ibmai', 'both']:
    ################ watsonx.ai section #####################
    parameters = {
        GenParams.DECODING_METHOD: "sample",
        GenParams.MAX_NEW_TOKENS: 100,
        GenParams.STOP_SEQUENCES: ["\n"],
        GenParams.TEMPERATURE:0.5,
        GenParams.REPETITION_PENALTY: 1.2,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }

    model = Model(
        model_id=model_id, 
        params=parameters, 
        credentials={
            "url": watsonx_api_url,
            "apikey": ibmai_key
        },
        project_id=ibmai_project)

    ########### Test Prompt ########
    print("Run Watsonx.ai testing Prompt")
    test_prompt = "Are you ready to work with me today, for our sentimental demo case for ?"
    test_output = model.generate_text(test_prompt)
    print("Test Prompt Output:", test_output)

    def get_predictions(prompt, ai_type):
        if ai_type == 'ibmai':
            response = model.generate_text(prompt)
            generated_text = response.strip().split("\n")[0]  # Assuming the first line contains the sentiment tag
            
            return generated_text

################ watsonx.ai section #####################
if not credentials.get('singlestore'):
    print("SingleStore credentials not provided. Skipping database operations.")
else:
    print("SingleStore credentials provided.")
    # Access SingleStore credentials from the credentials dictionary
    singlestore_creds = credentials['singlestore']
    print(f"Server: {singlestore_creds.get('hostname')}")
    
    ssl_config = {'ca': 'singlestore_bundle.pem'}

    try:
        connection = pymysql.connect(
            host=singlestore_creds.get('hostname'),
            port=int(singlestore_creds.get('port', 3306)),
            user=singlestore_creds.get('username'),
            password=singlestore_creds.get('password'),
            database=singlestore_creds.get('database'),
            ssl=ssl_config
        )

        # Create a cursor object here
        cursor = connection.cursor()
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        exit()

# Adjusted Main Logic for Sentiment Analysis
def fetch_comments():
    with connection.cursor() as cursor:
        cursor.execute(f"""
        SELECT c.comment_text, p.topic
        FROM comments AS c
        JOIN posts AS p ON c.post_shortcode = p.post_shortcode
        WHERE CHAR_LENGTH(c.comment_text) > 20 AND CHAR_LENGTH(c.comment_text) < 35
        LIMIT {comment_limit};
        """)
        return cursor.fetchall()
    
def load_prompt_template():
    with open('prompt_template.txt', 'r') as file:
        return file.read()

def load_openai_params():
    with open('openai_params.json', 'r') as file:
        return json.load(file)
    
def process_comments(comments, ai_type):
    results = []
    prompt_template = load_prompt_template()
    openai_params = load_openai_params()
    
    for comment, topic in tqdm(comments, desc="Processing comments"):
        # Perform sentiment analysis with TextBlob
        textblob_sentiment = analyze_sentiment_with_textblob(comment)
        
        # Initialize the AI sentiment as empty
        openai_sentiment = ""
        ibmai_sentiment = ""
        
        # Format the prompt with the actual manufacturer (topic) and comment
        prompt = prompt_template.format(manufacturer=topic, comment=comment)
        
        if ai_type == 'openai' or ai_type == 'both':
            # Make the call to the OpenAI model
            openai.api_key = openai_key
            response = openai.Completion.create(prompt=prompt, **openai_params)
            openai_sentiment = response.choices[0].text.strip().split("\n")[0]
        
        if ai_type == 'ibmai' or ai_type == 'both':
            # Make the call to the IBM Watson model
            ibmai_sentiment = get_predictions(prompt, 'ibmai')
        
        result = {
            'topic': topic,
            'comment': comment[:62] + '...' if len(comment) > 65 else comment,
            'textblob_sentiment': textblob_sentiment,
        }
        
        if ai_type == 'openai':
            result['openai_sentiment'] = openai_sentiment
        elif ai_type == 'ibmai':
            result['ibmai_sentiment'] = ibmai_sentiment
        elif ai_type == 'both':
            result['openai_sentiment'] = openai_sentiment
            result['ibmai_sentiment'] = ibmai_sentiment
        
        results.append(result)

    return pd.DataFrame(results)


def display_results(df):
    if not df.empty:
        # Preprocess DataFrame to truncate long text for each column
        for column in df.columns:
            df[column] = df[column].apply(lambda x: (x[:62] + '...') if isinstance(x, str) and len(x) > 65 else x)

        # Now, display the DataFrame
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    else:
        print("No data to display.")

# Execute
comments = fetch_comments()
results_df = process_comments(comments, args.aitype)
display_results(results_df)

# Cleanup
connection.close()

