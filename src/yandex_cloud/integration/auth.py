import requests
import json

def revoke_iam_token(iam_token: str):
    url = "https://iam.api.cloud.yandex.net/iam/v1/tokens:revoke"

    payload = {
        "iamToken": iam_token
    }

    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {iam_token}"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    return data

def create_iam_token(oauth_token: str):
    url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"

    payload = {
        "yandexPassportOauthToken": oauth_token
    }

    response = requests.post(url, json=payload)
    data = response.json()
    return data