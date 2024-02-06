#!/usr/bin/python3
import argparse
import os
import json
import getpass
from cryptography.fernet import Fernet, InvalidToken
import instaloader
import pymysql
import openai
import time
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Generate a key for encryption
key = Fernet.generate_key()
# Set up argument parser
parser = argparse.ArgumentParser(description='Download Instagram posts and comments.')
parser.add_argument('--user', type=str, help='Instagram username')
parser.add_argument('--password', type=str, help='Instagram password')
parser.add_argument('--master-password', type=str, help='Master password for the credential store')
parser.add_argument('--topic', '-t', type=str, help='Instagram channel name', required=True)
parser.add_argument('--posts', '-p', type=int, help='Number of posts to download', default=2)
parser.add_argument('--comments', '-c', type=int, help='Number of comments to fetch per post', default=40)
parser.add_argument('--skip_download', action='store_true', help='do not download instagram post and comments, only create vectors and store it into singlestore', required=False)
args = parser.parse_args()


SALT_FILE = 'salt.key'
CREDENTIALS_FILE = 'credentials.enc'

# Check if credential files exist
credentials_exist = os.path.exists(CREDENTIALS_FILE) and os.path.exists(SALT_FILE)

master_password = None
master_key = None
openai_credentials = {}
singlestore_credentials = {}

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

# Load OpenAI API key from environment variables
#openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_embeddings(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Waiting before retrying...")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return generate_embeddings(text)
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return None


# Function to read media as binary
def read_media_as_binary(media_path):
    with open(media_path, 'rb') as file:
        return file.read()
    
def get_instagram_credentials(master_key):
    """Retrieve Instagram credentials from the encrypted file."""
    try:
        with open(CREDENTIALS_FILE, 'rb') as file:
            encrypted_data = file.read()
        f = Fernet(master_key)
        decrypted_data = f.decrypt(encrypted_data)
        credentials = json.loads(decrypted_data.decode())
        return credentials.get('instagram', {'username': None, 'password': None})
    except Exception as e:
        print(f"Error retrieving credentials: {e}")
        return {'username': None, 'password': None}
    
# Function to insert data into SingleStoreDB
def insert_into_singlestore(posts, comments, media, singlestore_credentials):
    if not singlestore_credentials:
        print("SingleStore credentials not provided. Skipping database operations.")
        return
    else:
        print("SingleStore credentials provided. Starting import now.")

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
        
        with connection.cursor() as cursor:
            for post in posts:
                check_sql = "SELECT COUNT(*) FROM posts WHERE post_shortcode = %s"
                cursor.execute(check_sql, (post[1],))
                num_rows = cursor.fetchone()[0]
                if num_rows > 0:
                    print(f"Post {post[1]} for topic {post[0]} already exists in database. Skipping.")
                    continue
                else:
                    print(f"Post {post[1]} for topic {post[0]} will be imported.")

                    # Insert post
                    post_sql = "INSERT INTO posts (topic, post_shortcode, post_url, post_timestamp, meta_text) VALUES (%s, %s, %s, %s, %s)"
                    print("Executing:", cursor.mogrify(post_sql, post))
                    cursor.execute(post_sql, post)

                for comment in comments:
                    # Assuming 'comment' is a tuple like: (post_shortcode, username, text, timestamp, vector_json)
                    comment_text = json.dumps(comment[2])
                    #escaped_comment_text = comment_text #.replace('\n', '\\n') # Escape single quotes within comment_text
                    vector_json = json.dumps(comment[4])  # Convert the vector to a JSON string
                    comment_sql = f"""
                    INSERT INTO comments (post_shortcode, comment_username, comment_text, comment_timestamp, comment_vector)
                    VALUES ('{comment[0]}', '{comment[1]}', '{comment[2].replace("'", "''")}', '{comment[3]}', JSON_ARRAY_PACK('{vector_json}'))
                    """
                    print(f"Executing: {comment_sql}")
                    cursor.execute(comment_sql)

                # Insert media
                media_sql = "INSERT INTO media (post_shortcode, media_file, media_type) VALUES (%s, %s, %s)"
                for media_item in media:
                    print("Executing media insert for post_shortcode:", media_item[0], "; Media size:", len(media_item[1]), "bytes")
                    cursor.execute(media_sql, media_item)

        connection.commit()
    finally:
        connection.close()


# Initialize Instaloader with dirname pattern
L = instaloader.Instaloader(dirname_pattern=os.path.join(args.topic, "{shortcode}"),
                            download_videos=False,  # Do not download videos
                            download_video_thumbnails=False)  # Do not download video thumbnails

# Retrieve master password and derive master key
if credentials_exist:
        # Retrieve master password
    if args.master_password:
        master_password = args.master_password
    else:
        master_password = getpass.getpass("Enter master password: ")
    salt = get_salt()
    master_key = derive_key(master_password, salt)

    # Load OpenAI and SingleStore credentials from encrypted file
    openai_credentials = get_credentials(master_key, 'openai')
    singlestore_credentials = get_credentials(master_key, 'singlestore')

    # Set OpenAI API key
    openai.api_key = openai_credentials.get('api_key')

    # Display SingleStore credentials if retrieved
    if singlestore_credentials:
        print(f"SingleStore Host: {singlestore_credentials.get('hostname')}, "
              f"Port: {singlestore_credentials.get('port')}, "
              f"Database: {singlestore_credentials.get('database')}")
    else:
        print("Unable to retrieve SingleStore credentials.")

    # Load Instagram credentials
    instagram_credentials = get_instagram_credentials(master_key)
else:
    print("Credential files not found. Exiting.")
    exit(1)

try:
    if args.user and args.password:
        # Use provided credentials
        L.login(args.user, args.password)
    else:
        if instagram_credentials['username'] and instagram_credentials['password']:
            print("Successfully retrieved Instagram credentials.")
            #print(f"Username: {instagram_credentials['username']}, Password: {instagram_credentials['password']}")
            L.login(instagram_credentials['username'], instagram_credentials['password'])
        else:
            print("Unable to retrieve Instagram credentials.")
            exit(1)
except instaloader.exceptions.ConnectionException as e:
    print(f"Failed to log in to Instagram: {e}")
    exit(1)


# Load the profile
profile = instaloader.Profile.from_username(L.context, args.topic)

# Inside the loop where you process posts
for post in profile.get_posts():
    if args.posts <= 0:
        break

    # Define the directory for this specific post
    post_dir = os.path.join(args.topic, post.shortcode)

    # Initialize data for insertion
    post_data = []
    comment_data = []
    media_data = []

    # Example if clause using the --skip_download value
    if args.skip_download:
        print("Skipping download...")
    else:
        if os.path.exists(post_dir):
            print(f"Directory {post_dir} already exists. Skipping download...")
        else:
            print("Downloading Instagram posts and comments...")
            L.download_post(post, target=args.topic)

    # Read the metadata from the text file
    txt_files = [f for f in os.listdir(post_dir) if f.endswith('.txt')]
    meta_text = ''
    if txt_files:
        with open(os.path.join(post_dir, txt_files[0]), 'r') as file:
            meta_text = file.read()

    post_data.append((args.topic, post.shortcode, post.url, post.date_utc.isoformat(), meta_text))

    # Fetch and save comments
    for comment in post.get_comments():
        text = comment.text
        vector = generate_embeddings(text)
        if vector is not None:
            comment_data.append((post.shortcode, comment.owner.username, text, comment.created_at_utc.isoformat(), vector))

        # Append comment information for JSON file
        comments_json = {
            "username": comment.owner.username,
            "text": text,
            "created_at_utc": comment.created_at_utc.isoformat()
        }
        if len(comment_data) >= args.comments:
            break

    # Handle media files
    for file in os.listdir(post_dir):
        if file.endswith('.jpg') or file.endswith('.mp4'):
            media_path = os.path.join(post_dir, file)
            media_binary = read_media_as_binary(media_path)
            media_type = 'image' if file.endswith('.jpg') else 'video'
            media_data.append((post.shortcode, media_binary, media_type))

    # Save comments to JSON
    with open(os.path.join(post_dir, 'comments.json'), 'w') as f:
        json.dump(comments_json, f, indent=4)
        
    # Call the insert_into_singlestore function to insert data into the database
    insert_into_singlestore(post_data, comment_data, media_data, singlestore_credentials)

    args.posts -= 1
