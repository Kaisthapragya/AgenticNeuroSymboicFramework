from your_flask_app import app

# This is the entry point for the WSGI server
def application(environ, start_response):
    # Pass the incoming request to the Flask app
    return app(environ, start_response)

# If this script is run directly, start the WSGI server
if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    server = make_server('localhost', 8080, application)
    print("WSGI server running on http://localhost:8080")
    server.serve_forever()
