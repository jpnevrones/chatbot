from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# https://flask-socketio.readthedocs.io/en/latest/
# https://github.com/socketio/socket.io-client
question_list = []
global my_question, len_tracker
my_question = ''
len_tracker = 0

app = Flask(__name__)

app.config[ 'SECRET_KEY' ] = 'jsbcfsbfjefebw237u3gdbdc'
socketio = SocketIO( app )

@app.route( '/' )
def hello():
  return render_template( './ChatApp.html' )

def messageRecived():
  print( 'message was received!!!' )

@socketio.on( 'my event' )
def handle_my_custom_event( json ):
  print( 'recived my event: ' + str( json ) )
  socketio.emit( 'my response', json, callback=messageRecived )


def speak( msg ):
  socketio.emit( 'speakbot', msg, callback=messageRecived )

@socketio.on( 'speak event' )
def handle_my_custom_event( json ):
  global my_question, len_tracker

  respones_msg =  json['message']
  end = len(respones_msg)
  new_resp = respones_msg[len_tracker:end]
  if my_question != new_resp:
    if new_resp != "":
      my_question = new_resp

      print('transcript recived: ' + my_question)
      if my_question != '':
          len_tracker = len_tracker + len(my_question)
          socketio.emit( 'speech rec', my_question, callback=messageRecived )
          resp = "Will respond when you will put my brain back"
          socketio.emit('bot resp', resp, callback=messageRecived)

      #speak("I can speak now")




# @app.route('/speech')
# def speech():
#   if request.method == 'POST':
#     pass

# @socketio.on( 'my event' )
# def handle_my_custom_event( json ):
#   print( 'recived my event: ' + str( json ) )
#   socketio.emit( 'my response', json, callback=messageRecived )

if __name__ == '__main__':
  socketio.run( app, debug = True )