#!/usr/bin/python3
import os
import json
import pymysql
import openai
import argparse
import getpass
import pandas as pd
from tabulate import tabulate
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import platform  # Import platform module to clear the screen


# Set up argument parser
parser = argparse.ArgumentParser(description='Fetch Singlestore and API AI Key')
parser.add_argument('--master-password', type=str, help='Master password for the credential store')
args = parser.parse_args()

SALT_FILE = 'salt.key'
CREDENTIALS_FILE = 'credentials.enc'

# Check if credential files exist
credentials_exist = os.path.exists(CREDENTIALS_FILE) and os.path.exists(SALT_FILE)

master_password = None
master_key = None
openai_credentials = {}
singlestore_credentials = {}

def clear_screen():
    """Clear the terminal screen based on the operating system."""
    if platform.system() == 'Windows':
        os.system('cls')  # For Windows
    else:
        os.system('clear')  # For Linux/Unix

def get_salt():
    """Retrieve or generate a salt for key derivation."""
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, 'rb') as file:
            return file.read()
    else:
        salt = os.urandom(16)
        with open(SALT_FILE, 'wb') as file:
            file.write(salt)
        return salt

def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a key from a password and salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

# New function to retrieve SingleStore and OpenAI credentials
def get_credentials(master_key, service_type):
    """Retrieve credentials for a specific service from the encrypted file."""
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

# Custom function to truncate or insert line breaks
def format_text(text, max_length=200):
    return text if len(text) <= max_length else text[:max_length] + '...'

# Apply the function to your DataFrame
def format_dataframe(df):
    for col in df.columns:
        if df[col].dtype == object:  # Apply only to string columns
            df[col] = df[col].apply(lambda x: format_text(x, 200))
    return df

# Function to get embeddings using OpenAI
def get_embeddings(text):
    response = openai.Embedding.create(input=[text], engine="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Retrieve master password and derive master key
if credentials_exist:
    if args.master_password:
        master_password = args.master_password
    else:
        master_password = getpass.getpass("Enter master password: ")
    salt = get_salt()
    master_key = derive_key(master_password, salt)

# Get OpenAI and SingleStore credentials from the encrypted file
openai_credentials = get_credentials(master_key, 'openai')
singlestore_credentials = get_credentials(master_key, 'singlestore')

# Set OpenAI API key
openai.api_key = openai_credentials.get('api_key')

if not singlestore_credentials:
    print("SingleStore credentials not provided. Skipping database operations.")
    exit()  # Exit the script if credentials are not provided
else:
    print("SingleStore credentials provided.")

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
        
# Function to fetch top 5 results grouped by topic
def fetch_top__results_grouped_by_topic(search_string):
    search_embedding = json.dumps(get_embeddings(search_string))
    
    query = f"""
    SELECT
        a.topic,
        SUM(DOT_PRODUCT(c.comment_vector, JSON_ARRAY_PACK('{search_embedding}'))) / COUNT(c.comment_text) AS average_score_per_comment,
        COUNT(DISTINCT c.comment_text) AS distinct_comment_count,
        COUNT(DISTINCT a.post_shortcode) AS distinct_post_count
    FROM
        posts a
    JOIN
        comments c ON a.post_shortcode = c.post_shortcode
    GROUP BY a.topic
    ORDER BY average_score_per_comment DESC
    LIMIT 20;
    """
    #print(query)

    try:
        cursor.execute(query)
        results = cursor.fetchall()

        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=['Topic', 'Score','Count_Comments', 'Count_Posts'])
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


# Function to fetch results based on search string
def fetch_results(search_string):
    search_embedding = json.dumps(get_embeddings(search_string))

    query = f"""
    SELECT
        a.topic,
        c.comment_text,
        DOT_PRODUCT(c.comment_vector, JSON_ARRAY_PACK('{search_embedding}')) AS score,
        c.post_shortcode
    FROM
        comments c
    JOIN
        posts a ON c.post_shortcode = a.post_shortcode
    ORDER BY score DESC
    LIMIT 30;
    """

    #print(query)
    try:
        cursor.execute(query)
        results = cursor.fetchall()

        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=['Topic', 'Comment Text', 'Score', 'Post Shortcode'])
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error
    
# Main interaction loop
while True:
    clear_screen()  # Clear the screen before each interaction
    print("Choose an option:")
    print("1. Search for comments")
    print("2. Get Top s")
    print("3. Exit")
    
    option = input("Enter your choice (1/2/3): ")

    if option == '1':
        search_string = input('Please enter a search string comments: ')

        # Call the function to fetch results
        results_df = fetch_results(search_string)

        # Format the DataFrame
        formatted_df = format_dataframe(results_df)

        # Display the results DataFrame with borders and lines using tabulate
        print(tabulate(formatted_df, headers='keys', tablefmt='grid', showindex=False))
        input('Press Enter to continue...')  # Wait for user input to continue
        
    elif option == '2':
        search_string = input('Please enter a search string topic: ')
        # Call the function to fetch top 5 results grouped by topic
        top__results_df = fetch_top__results_grouped_by_topic(search_string)
        formatted_df = format_dataframe(top__results_df)

        # Display the top 5 results grouped by topic
        print(tabulate(top__results_df, headers='keys', tablefmt='grid', showindex=False))
        input('Press Enter to continue...')
        
    elif option == '3':
        break  # Exit the loop if the user chooses to exit

# Close the database connection
if cursor:
    cursor.close()
if connection:
    connection.close()
