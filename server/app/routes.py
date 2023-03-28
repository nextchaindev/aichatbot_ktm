from flask import session, redirect, url_for, render_template, request, Blueprint

main = Blueprint('main', __name__)


@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('main.chat'))
    return render_template('index.html')

@main.route('/chat')
def chat():
    return render_template('chat.html', name="name", room="room")