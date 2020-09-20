# Disaster Response Pipeline

## Introduction
The idea for this project is to leverage data from a disaster pipeline, preprocess it, store it into a database, create a predictive model, and demonstrate those predictions in a flask app. These predictions are based on real messages that were sent during disaster events from [Figure Eight](https://appen.com/). The idea here is to create a practical predictive model, that could be used to help people.

## File Structure
* app
	* | - template
	* | |- master.html # main page of web app
	* | |- go.html # classification result page of web app
	* |- run.py # Flask file that runs app

* data
	* |- disaster_categories.csv # data to process
	* |- disaster_messages.csv # data to process
	* |- process_data.py
	* |- InsertDatabaseName.db # database to save clean data to

* models
	* |- train_classifier.py
	* |- classifier.pkl # saved model

README.md

## ETL (Extract, Transform, and Load)
### Overview
In the folder marked `data/` we can see csv files, a database, and a python script. The python script reads in these two csv files, cleans them, and joins them by ID. The data cleaning results in a larger dataframe, with a 1 or a 0, if any of the categories applies to the message column. This cleaned and joined data frame is then uploaded and stored as a database: `DisasterResponse.db`.

### How to use
`cd` into the `data/` folder, and run this command: `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`.

## Machine Learning
### Overview
In the folder name `models/` there is a python script, which reads in the database from the ETL step, and outputs a "pickled" machine learning model. This model has been trained on the dataset, and run through a machine learning "pipeline". This pipeline takes in 3 inputs: a vectorized version of the message (using TF-IDF), whether or not the message starts with a verb, and the length of the message. These inputs are piped into a Multiple Output classifier, which leverages a Random Forest classifier. This pipeline is then tuned to hyperparameters using grid search.

### How to use
`cd` into the `models/` folder and run this command: `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`.

## Deployment
### Overview
In the folder named `app/` there is a python script called `run.py`, which contains the loading of the data, model, and routing of the flask app. The flask app presents an input field for classifying messages, based on the predictive model, as well as 3 visualizations on the homepage:

1. The distribuiton of genres in the messages
2. The distribution of categories in the messages
3. The distribution of the average length of the messages, in each of the categories

### How to use
`cd` into the `app/` folder and run this command: `python run.py`, and the flask app will be hosted at `http://0.0.0.0:3001/`

