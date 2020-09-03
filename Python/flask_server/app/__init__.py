from flask import Flask, flash, request, abort, jsonify, send_from_directory, redirect, url_for, render_template, \
    send_file
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.run(debug=True)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

from app import routes
