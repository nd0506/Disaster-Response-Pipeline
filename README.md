# Disaster-Response-Pipeline
## Overview
This project aims to build a machine learning model to classify messages that are sent during the disasters in order to send them to an appropriate disaster relief agency.

The project is using real messages that were sent during the disaster events for training the model. The datasets are provided by [Appen](https://www.appen.com/)

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Repository structure
### data
* `disaster_messages.csv` : Messages that were sent during the disasters
* `disaster_categories.csv`: Category of each messages. There are 36 categories in total. Messages and categories data are matched by common id.
* `process_data.py`: File that contains codes for loading, cleaning and saving messages and categories data
* `DisasterResponse.db`: Database to save clean data to

### model
* `train_classifier.py`: File that runs the model
### app
* `run.py`: Flask file that runs app
  
* **templates**
  * `master.html`: Main page of web app
  * `go.html` : Classification result of web app

### README.md
Project instruction

## How to run the codes:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

## Acknowledgements
* Data credits: [Appen](https://www.appen.com/)
* Project instruction: Udacity instructors and mentors for Data Scientist Program
   
