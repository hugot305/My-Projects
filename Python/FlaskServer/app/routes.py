import os
import pandas as pd
from app import app
from app import flash, jsonify, send_from_directory, abort, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@app.route('/files/upload', methods=['GET', 'POST'])
def upload_file():
    """Upload a file."""
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']

            # if user does not select file, browser also
            # submit an empty part without filename

            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('list_files'))
    except Exception as e:
        return str(e)

    return render_template('upload.html')


@app.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(path):
            files.append(filename)
    return render_template('files.html', result=files)


@app.route("/files/download/<path:path>")
def download_file(path):
    """Download a file."""
    try:
        return send_file(os.path.join("..//" + app.config['UPLOAD_FOLDER'], path), as_attachment=True)
    except Exception as e:
        return str(e)


@app.route("/files/view/<path:path>/<int:page>")
def view_file(path, page):
    """View file's content."""
    try:
        print(path)
        data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], path))
        data['state'] = data['state'].fillna('Blank')
        rows = data.loc[(page * 10):(((page * 10) + 10) - 1)]
        previous_page = page - 1
        if previous_page < 0:
            previous_page = 0
        next_page = page + 1
        last_page = (int(len(data.index) / 10))
        if next_page > last_page:
            next_page = last_page
        return render_template('view.html', result=rows, file=path, current_page=page, previous_page=previous_page,
                               next_page=next_page, last_page=last_page, total_records=len(data.index))
    except Exception as e:
        return str(e)


@app.route("/files/count/<path:path>")
def year_count(path):
    """View file's content."""
    try:
        data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], path))
        data['year'] = pd.DatetimeIndex(data['date']).year
        data = data.groupby('year').filter(lambda x: len(x) > 1)
        count = data['year'].value_counts()

        # Select the values where the count is less than 3 (or 5 if you like)
        to_remove = count[count < 2].index

        # Keep rows where the city column is not in to_remove
        data = data[~data.year.isin(to_remove)]

        count = data['year'].value_counts()

        return render_template('count.html', result=count.iteritems(), records=len(count.index))
    except Exception as e:
        return str(e)
