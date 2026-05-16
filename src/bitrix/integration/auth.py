import requests

def bot_register(webhook, bot_token, bot_code, bot_name, bot_work_position):
    url = webhook + 'imbot.v2.Bot.register'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "botToken": bot_token,
        "fields": {
            "code": bot_code,
            "properties": {
                "name": bot_name, 
                "workPosition": bot_work_position
            },
            "eventMode": "fetch",
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def bot_unregister(webhook, bot_id, bot_token):
    url = webhook + 'imbot.v2.Bot.unregister'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "botId": int(bot_id),
        "botToken": bot_token
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def get_bot_list(webhook, bot_token):
    url = webhook + 'imbot.v2.Bot.list'
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "botToken": bot_token,
        "filter":{"type":"bot"},
        "limit":10
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()
