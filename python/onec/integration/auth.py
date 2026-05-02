import base64

def create_auth_header(username, password):
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
    auth_header = f"Basic {encoded_credentials}"
    return auth_header

