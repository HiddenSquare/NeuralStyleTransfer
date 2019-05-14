from flask import Flask
from flask_restplus import Api
from waitress import serve

class App:
    """Main entrypoint for REST API.
    """
    def __init__(self):

        self.api = Api(
            title='Style transfer api',
            version='1.0',
            description='A description',
        )

        from endpoints import api
        self.api.add_namespace(api, path="/api")

    def run(self):
        """Start Flask web server"""
        app = Flask(__name__)
        self.api.init_app(app)

        serve(app, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # The api type is provided when called from docker
    App().run()