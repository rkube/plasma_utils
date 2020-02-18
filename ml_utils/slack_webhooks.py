# -*- Encoding: UTF-8 -*-

webhook_url = "https://hooks.slack.com/services/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"

def post_image(filename, token, channels):
    f = {'file': (filename, open(filename, 'rb'), 'image/png', {'Expires':'0'})}
    response = requests.post(url='https://slack.com/api/files.upload', data=
       {'token': token, 'channels': channels, 'media': f}, 
       headers={'Accept': 'application/json'}, files=f)
    return response.text


# End of file slack_webhooks.py