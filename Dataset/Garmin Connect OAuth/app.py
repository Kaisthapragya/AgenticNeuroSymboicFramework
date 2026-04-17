from flask import Flask, render_template, request, redirect, session
import requests
from requests_oauthlib import OAuth1

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secret key for session management

# Replace these with your actual values
consumer_key = ""
consumer_secret = ""

request_token = ""
request_token_secret = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_user_token', methods=['POST'])
def get_user_token():
    global request_token, request_token_secret

    # Step 1: Acquire Unauthorized Request Token
    request_token_url = "https://connectapi.garmin.com/oauth-service/oauth/request_token"
    auth = OAuth1(consumer_key, client_secret=consumer_secret)

    response = requests.post(request_token_url, auth=auth)

    if response.status_code == 200:
        request_token_data = dict(param.split("=") for param in response.text.split("&"))
        request_token = request_token_data["oauth_token"]
        request_token_secret = request_token_data["oauth_token_secret"]
        print(request_token)
        print(request_token_secret)

        # Store request tokens in the session for later use
        session['request_token'] = request_token
        session['request_token_secret'] = request_token_secret

        # Step 2: Authorization of the Request Token
        authorization_url = f"https://connect.garmin.com/oauthConfirm?oauth_token={request_token}"
        
        # Redirect the user to the Garmin authorization page
        return redirect(authorization_url)

    return "Error acquiring Request Token"

@app.route('/oauth_callback')
def oauth_callback():
    global request_token, request_token_secret

    oauth_verifier = request.args.get('oauth_verifier')

    # Retrieve request tokens from the session
    request_token = session.get('request_token')
    request_token_secret = session.get('request_token_secret')

    # Step 3: Acquire User Access Token
    access_token_url = "https://connectapi.garmin.com/oauth-service/oauth/access_token"
    auth = OAuth1(consumer_key,
                  client_secret=consumer_secret,
                  resource_owner_key=request_token,
                  resource_owner_secret=request_token_secret,
                  verifier=oauth_verifier)

    response = requests.post(access_token_url, auth=auth)

    if response.status_code == 200:
        access_token_data = dict(param.split("=") for param in response.text.split("&"))
        user_access_token = access_token_data["oauth_token"]
        user_access_token_secret = access_token_data["oauth_token_secret"]

        # Now, you can use the user_access_token and user_access_token_secret to make authorized requests
        # to Garmin API on behalf of the user. You may store these tokens in your database for future use.

        # For example, you can make a request to get the user's profile data
        user_profile_url = "https://connectapi.garmin.com/user-profile-service/1/user/profile"
        auth = OAuth1(consumer_key,
                      client_secret=consumer_secret,
                      resource_owner_key=user_access_token,
                      resource_owner_secret=user_access_token_secret)

        profile_response = requests.get(user_profile_url, auth=auth)

        if profile_response.status_code == 200:
            # Process the user profile data as needed
            user_profile_data = profile_response.json()
            return f"User Profile Data: {user_profile_data}"

    return "Error acquiring User Access Token"

if __name__ == '__main__':
    app.run(debug=True)
