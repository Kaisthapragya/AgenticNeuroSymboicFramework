
# Garmin Connect OAuth

This project demonstrates how to authenticate and fetch data from Garmin Connect using OAuth. The application uses Flask to run a simple web server that handles OAuth authentication and data retrieval.

## Prerequisites

Before you start, ensure you have the following:
- Python 3.6 or higher
- Flask
- Requests

You can install the necessary Python libraries using:
```bash
pip install flask requests
```

## Configuration

### Token Setup
1. Navigate to the Garmin Connect developer portal and create a new application to receive your `client_id` and `client_secret`.
2. Use the `verify.py` script to enter your `client_id` and `client_secret`. This script will guide you through obtaining your user authorization by directing you to the Garmin Connect login page to authenticate and grant access. 
3. After authentication, Garmin will redirect you back with an authorization code. The `verify.py` script will automatically exchange this code for a `user_token` and `user_secret`.
4. The script will then automatically insert these tokens into the appropriate configuration places within your application, ensuring that your credentials are correctly set up for accessing the Garmin Connect API.

### Environment Variables
Set the following environment variables:
- `CLIENT_ID`: Your Garmin Connect application client ID.
- `CLIENT_SECRET`: Your Garmin Connect application client secret.

## Running the Application

1. Start the application server by running:
   ```bash
   python WSGI.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`.
3. Click the **Get User Token** button to authenticate with Garmin Connect and retrieve your access token.

## Usage

After authenticating, you can use the access token obtained to make requests to the Garmin Connect APIs according to your application's requirements. Open the `verify.py` file, insert your tokens and verifier, and then run `verify.py` to add the user to your app.

## Additional Information

- **verify.py**: This script is critical for setting up your Garmin Connect credentials. It manages the initial OAuth flow, including the exchange of authorization codes for access tokens. It's designed to securely handle your tokens and automatically configure them within the application.
- **app.py**: The main Flask application script.
- **WSGI.py**: Script to start the WSGI server for the Flask app.
- **index.html**: Contains the HTML form used to start the OAuth flow.

For more information on Garmin Connect API and OAuth flow, refer to the Garmin Connect developer documentation.
