#!/usr/bin/python3
import argparse
import os
from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
import getpass

# Constants
CREDENTIALS_FILE = 'credentials.enc'
SALT_FILE = 'salt.key'

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


def write_encrypted_data(data, key, filename=CREDENTIALS_FILE):
    """Encrypt and write data to a file."""
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    with open(filename, 'wb') as file:
        file.write(encrypted_data)

def read_encrypted_data(key, filename=CREDENTIALS_FILE):
    """Read and decrypt data from a file."""
    f = Fernet(key)
    with open(filename, 'rb') as file:
        encrypted_data = file.read()
    return f.decrypt(encrypted_data).decode()

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

def main():
    parser = argparse.ArgumentParser(description='Manage service credentials')
    parser.add_argument('-create', action='store_true', help='Create new credentials')
    parser.add_argument('-getpass', action='store_true', help='Retrieve credentials')
    parser.add_argument('-type', type=str, choices=['instagram', 'singlestore', 'openai', 'ibmai'], help='Type of service')

    args = parser.parse_args()
    master_password = getpass.getpass("Enter master password: ")
    salt = get_salt()
    master_key = derive_key(master_password, salt)
    credentials = {}

    try:
        if os.path.exists(CREDENTIALS_FILE):
            encrypted_data = read_encrypted_data(master_key)
            credentials = json.loads(encrypted_data)

        if args.create and args.type:
            if args.type == 'singlestore':
                hostname = input("Enter hostname for SingleStore: ")
                port = input("Enter port for SingleStore: ")
                database = input("Enter database name for SingleStore: ")
                username = input("Enter username for SingleStore: ")
                password = getpass.getpass("Enter password for SingleStore: ")
                credentials[args.type] = {'hostname': hostname, 'port': port, 'database': database, 'username': username, 'password': password}
            elif args.type in ['openai', 'ibmai']:
                api_key = getpass.getpass(f"Enter API key for {args.type}: ")
                credentials[args.type] = {'api_key': api_key}
            else:
                username = input(f"Enter username for {args.type}: ")
                password = getpass.getpass(f"Enter password for {args.type}: ")
                credentials[args.type] = {'username': username, 'password': password}

            write_encrypted_data(json.dumps(credentials), master_key)
            print(f"Credentials for {args.type} created.")

        elif args.getpass and args.type:
            if args.type in credentials:
                credential = credentials[args.type]
                if 'api_key' in credential:
                    print(f"API Key: {credential['api_key']}")
                else:
                    print(f"Username: {credential.get('username', '')}")
                    print(f"Password: {credential.get('password', '')}")
                    if args.type == 'singlestore':
                        print(f"Hostname: {credential.get('hostname', '')}")
                        print(f"Port: {credential.get('port', '')}")
                        print(f"Database: {credential.get('database', '')}")
            else:
                print(f"No credentials found for {args.type}")

    except InvalidToken:
        print("You entered a wrong master password")

if __name__ == "__main__":
    main()