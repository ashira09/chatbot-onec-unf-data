import requests

def fetch_message(webhook, bot_id, bot_token, limit, offset):
    url = webhook + 'imbot.v2.Event.get'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "botId": int(bot_id),
        "botToken": bot_token,
        "limit": limit,
        "offset": offset
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def send_message(webhook, bot_id, bot_token, dialog_id, message):
    url = webhook + 'imbot.v2.Chat.Message.send'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "botId": int(bot_id),
        "botToken": bot_token,
        "dialogId": dialog_id,
        "fields": {
            "message": message
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()