from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO(logger=True, engineio_logger=True)

def create_app(debug=False):

    app = Flask(__name__)
    app.debug = debug
    app.config['SECRET_KEY'] = 'hard to guess...'

    socketio.init_app(app)

    from app.events import ChatNamespace
    socketio.on_namespace(ChatNamespace('/chat'))

    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app

