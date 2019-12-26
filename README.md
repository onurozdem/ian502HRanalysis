# IAN502 - Data Programming Lesson  Term Project <br> HR Data Analysis 

### Project Members
- Melek Elmas
- Fırat Varol 
- Safa Can Demir 
- Onur Özdemir

<br>

### Project Goal
We are aiming to predict employee attrition as our first goal. For our secondary objectives we can try to segment the employees for certain Education levels or regarding their demographical data.
In addition, our final goal is to develop a model over the data set and to provide this model to user with an interface. Users will be able to input the data they want to predict through the user interface and will be able to access the model outputs along with the prediction result. After each new data prediction, new incoming data adding to the learning data and model update. Addition, we will give chance to user to compare with a few algorithm on interface. 
Finally, our goal is to develop a dynamic, self-updating and living project.

<br>

### Dataset to be Used
https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset#WA_Fn-UseC_-HR-Employee-Attrition.csv
This is a synthetic dataset created by the IBM Data Scientists with a main purpose of demonstrating the analytics tool for employee attrition. So, it's useful for testing our own models that require employee data but could be useless for drawing any real-world conclusions.
There are 35 columns and 1470 rows in the data. 26 of the columns are Numerical and 9 are Nominal marked with (*). Below is the list of the columns. 

<br>

### Data Processing Methodology
For employee attrition and classification, we will use Logistic Regression, Random Forest, Adaboost, Decisio Tree, LDA, Naive Bayes, ANN, Random Forest, SVM and KNN algorithms on data analysis phase and we will give chance to user to compare of these algorithms on interface.  

<br>

### Project Methodology
CRISP-DM

<br>

### Project Infrastructure
We will use Python for model development and front-end services. We will develop with Django framework for the front side. We will keep the data on SQLite. Technology stack:
•	Python
•	Django 
•	SQLite
•	Google Cloud Compute Engine

<br>

##Project Code Management
Codes of project for versioning and management of code arrangements between team members will be stored in a repo on github.
Github: https://github.com/onurozdem/ian502HRanalysis

<br>

## Template Source
https://www.free-css.com/free-css-templates/page246/greatness

## Graph Source
https://d3js.org/

# Installation & Running Application
Python3.6 is required for stable running.

### Library Installation
You can find requirements file on project's root directory. Run command at below for library insallation.

    $ pip install -r requirements.txt 

### Run Applicaiton

    $ python manage.py runserver 
