import base64
import getpass
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

def adjust_key(key):
    """Ensure the key is exactly 32 bytes long by padding or trimming."""
    key = key.encode()  # Convert string to bytes
    return key.ljust(32, b' ')[:32]  # Pad with spaces or trim

def decrypt_file(encrypted_path):
    # Read the AES key securely from the user
    raw_key = getpass.getpass("Enter decryption key: ")
    key = adjust_key(raw_key)  # Adjust the key to 32 bytes

    # Read the encrypted file
    with open(encrypted_path, "rb") as f:
        data = f.read()

    # Extract IV and encrypted code
    iv = data[:16]  # First 16 bytes is IV
    encrypted_code = data[16:]  # Remaining is the encrypted content

    # Initialize AES cipher in CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt and remove padding
    decrypted_code = unpad(cipher.decrypt(encrypted_code), AES.block_size).decode()

    # Create a new file for the decrypted code
    decrypted_path = encrypted_path.replace(".enc", "_decrypted.py")
    with open(decrypted_path, "w") as f:
        f.write(decrypted_code)

    print(f"Decrypted file saved as {decrypted_path}")

# Example Usage
encrypted_path = input("Enter the path of the encrypted file: ")
decrypt_file(encrypted_path)
