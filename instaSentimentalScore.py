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
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from textblob import TextBlob
import nltk
import re
import csv
import datetime
from langchain.prompts import PromptTemplate
from genai.credentials import Credentials
from dotenv import load_dotenv
# Using Generative AI Library
from genai.model import Model
from genai.schemas import GenerateParams
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Suppress all warnings
import warnings
import seaborn as sns
from wordcloud import WordCloud




nltk.download('punkt')




# Argument Parsing
parser = argparse.ArgumentParser(description='Sentiment Analysis with Secure Credentials')
parser.add_argument('--master-password', type=str, help='Master password for the credential store')
parser.add_argument('-aitype', choices=['openai', 'ibmai', 'both'], required=True, help='AI type for sentiment analysis')
args = parser.parse_args()

SALT_FILE = 'salt.key'
CREDENTIALS_FILE = 'credentials.enc'

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
if os.path.exists(CREDENTIALS_FILE) and os.path.exists(SALT_FILE):
    master_password = args.master_password or getpass("Enter master password: ")
    salt = get_salt()
    master_key = derive_key(master_password, salt)
    openai_credentials = get_credentials(master_key, 'openai')
    ibmai_credentials = get_credentials(master_key, 'ibmai')
    singlestore_credentials = get_credentials(master_key, 'singlestore')
else:
    print("Credential files not found.")
    exit()
    
# p-2+IBUPzgoeLjgBztO/lLZHHg==;TAq/ujHl/IiSpnDt3iCIBA==:XBiA2TCiUDDouougZKeuiQHPVdUKlAprRkzmg7H5BjoxEt2NjU6IO1voLKcaoQsorWBGt6xRYRsN76DyJw6wpPMI6i4b+0+vXg==
#api_url = "https://bam-api.res.ibm.com/v2/text/generation?version=2024-01-10"
api_url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"

if not ibmai_credentials:
    print("watsonx.ai credentials not provided.")
    exit()  # Exit the script if credentials are not provided
else:
    print("watsonx.ai credentials provided.")
    ibmai_key = ibmai_credentials.get('api_key')
    print(f"watsonx.ai API key: {ibmai_key}")
    print(f"watsonx.ai URL:" + api_url )

if not openai_credentials:
    print("watsonx.ai credentials not provided.")
    exit()  # Exit the script if credentials are not provided
else:
    print("openai credentials provided.")
    openai_key = openai_credentials.get('api_key')
    print(f"openai API key: {openai_key}")

################ watsonx.ai section #####################

prompt_string="""
You are a Social Media Analyst! You help in making the comments that user post easier to analyze by categorizing them. Categorize the following comment under post on Instagram of the official Mercedes Benz account into one of the available tags in list_of_tags.

list_of_tags = [
"positive sentiments towards the brand Mercedes",
"positive sentiments towards the mentioned model",
"negative sentiments towards the brand Mercedes",
"negative sentiments towards the mentioned model",
"questions or inquiries",
"personal experience or stories with the brand Mercedes",
"personal experience or stories with the mentioned model",
"political statements",
"geographical statements",
"miscellaneous or unclear"
]

COMMENT: This is so amazing! It's my dream car
TAG: positive sentiments towards the mentioned model

COMMENT: Mein größter Traum ist eines Tages für Mercedes zu arbeiten
TAG: personal experience or stories with the brand Mercedes

COMMENT: TF has happened to MB design team.
TAG: negative sentiments towards the brand Mercedes

COMMENT: Amazing! Way better than 5 series.
TAG: positive sentiments towards the mentioned model

COMMENT: Expect more from Mercedes my truck has been getting serviced for 6 months with no one responding!!!!!!
TAG: questions or inquiries

COMMENT: Why Mercedes why????
TAG: negative sentiments towards the brand Mercedes

COMMENT:{comment}
"""


################ watsonx.ai section #####################


if not singlestore_credentials:
    print("SingleStore credentials not provided. Skipping database operations.")
    exit()  # Exit the script if credentials are not provided
else:
    print("SingleStore credentials provided.")
    print(f"Server: " + singlestore_credentials.get('hostname'))

    ssl_config = {'ca': 'singlestore_bundle.pem'}

    try:
        connection = pymysql.connect(
            host=singlestore_credentials.get('hostname'),
            port=int(singlestore_credentials.get('port', 3306)),
            user=singlestore_credentials.get('username'),
            password=singlestore_credentials.get('password'),
            database=singlestore_credentials.get('database'),
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
        cursor.execute("""
        SELECT c.comment_text, p.topic
        FROM comments AS c
        JOIN posts AS p ON c.post_shortcode = p.post_shortcode
        WHERE CHAR_LENGTH(c.comment_text) > 20 AND CHAR_LENGTH(c.comment_text) < 35
        LIMIT 30;
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
    
    for comment, topic in comments:
        # Perform sentiment analysis with TextBlob
        textblob_sentiment = analyze_sentiment_with_textblob(comment)
        
        # Initialize the AI sentiment as empty
        ai_sentiment = ""
        
        if ai_type in ['openai', 'both']:
            # Format the prompt with the actual manufacturer (topic) and comment
            prompt = prompt_template.format(manufacturer=topic, comment=comment)
            # Make the call to the AI model
            response = openai.Completion.create(prompt=prompt, **openai_params)
            # Extract the sentiment response (TAG)
            ai_sentiment = response.choices[0].text.strip().split("\n")[0]  # Correctly taking the first TAG response

        
        result = {
            'topic': topic,
            'comment': comment[:62] + '...' if len(comment) > 65 else comment,  # Truncate long comments
            'textblob_sentiment': textblob_sentiment,  # Sentiment from TextBlob
            'openai_sentiment': ai_sentiment  # Sentiment from OpenAI
        }
        
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

