# **CSV Web Application** 

---

**CSV Web Application**

In this project my goal was to write a small web application that will allow the user to upload, download and view csv files. This project has been tested on Windows 10 and Linux Ubuntu.

---
### Dependencies
The only special libraries used in this project are: Flask,  Flask-Bootstrap and Pandas.

You can install the dependencies like this:
```
	pip install flask
	pip install flask-bootstrap
	pip install pandas
```
### To run the project
For Linux:
```
	export FLASK_APP=server.py
	flask run
```
For Windows:
```
	set FLASK_APP=server.py
	flask run
```

### Limits
The maximum file size is 10 megabytes. If a larger file is transmitted, Flask will raise a RequestEntityTooLarge exception.

When using the local development server, you may get a connection reset error instead of a 413 response. You will get the correct status response when running the app with a production WSGI server.

---
### Requirements

The application should have the ability to:
- Upload a CSV file
- List uploaded CSV files
- Download the previously uploaded CSV file
- Display the CSV content showing at least all column headers and content
- Provide statistics on the number of people with the same year in the “date” field.

#### Uploading files
![alternate text](images/upload.png "Uploading file")

#### File list
![alternate text](images/file_list.png "File list")

### Number of people with the same year in the “date” field
![alternate text](images/count.png "Counting members with same year")

#### View file content
![alternate text](images/view.png "Viewing file content")

#### Download file
![alternate text](images/download.png "Downloading file")


