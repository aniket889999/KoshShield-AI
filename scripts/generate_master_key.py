from base64 import urlsafe_b64encode
from secrets import token_bytes


if __name__ == "__main__":
    print(urlsafe_b64encode(token_bytes(32)).decode("ascii"))
