import requests

def bot_register(webhook, bot_token, bot_code, bot_name, bot_work_position, bot_event_mode, webhook_url):
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
            "eventMode": bot_event_mode,
            "webhookUrl": webhook_url
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