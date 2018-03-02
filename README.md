# Machine Learning Engineer Nanodegree (by Udacity)

# Capstone Project

I am making my project - along with the code - available online in case it is useful for other students or people who are getting interested in Machine Learning. Questions and suggestions are welcome. 

## Predicting H-1B visa eligibility in the U.S. 

Every year hundreds of thousands of international workers apply for H-1B non-immigrant visas in the United States. In order to be able to qualify for worker H1-B visas, a person needs to have a job offer from a U.S. based company. This is also the kind of visa usually requested by international students pursuing higher education in the country. This study aims to train a classifier based on features of the dataset to be able to predict whether a given request would be granted eligibility to the H-1B program. Given the number of people requesting visas every year - and the likelihood to increase over the next years despite political pressure - it would be interesting to analyze some of the existing data and provide a model that could help to understand successful over non-successful applications. This is handled as a multi-class classification problem as we have to identify one among different solutions, however the number of outputs used will be discussed given the fact that some of these results are more influenced by external factors and do not have a significant impact on the results. To approach this problem three different classifiers are trained and compared to identify the five most important features to tackle this problem. While prevailing wage is the highest weighted features as expected, part-time positions weight stronger than expected and the worksite does not necessarily affects the outcome of the application. Finally, a Logistic Regression classifier proved to be the best option among those analyzed to process this data considering time needed to train and predict as well as output produced.

Code available on Kaggle notebook too: 

https://www.kaggle.com/elraphabr/predicting-outcome-for-h-1b-eligibility-in-the-us 

# Also, here is a quick guide on how to get your jupyter notebook running on AWS: 

# —> On Local Machine: 

## Select the instance in AWS 

### Configure security group 
SSH TCP 22 0.0.0.0/0
Custom TCP Rule TCP 8888 0.0.0.0/0

## Create a new pair of keys and keep the private secure 
AWS_private_key.pem

## Adjust the access privileges for your key 
chmod 400 AWS_private_key.pem
ssh-add AWS_private_key.pem

## Connect via SSH using your IP 
ssh -i "AWS_private_key.pem" ec2-user@ec2-[IP].eu-central-1.compute.amazonaws.com 

# —> On AWS Machine 

## Updates
sudo yum update

## Download and Install Anaconda

wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
sh Anaconda2-5.0.1-Linux-x86_64.sh

## Set the PATH to execute anaconda’s commands (import numpy) and kill jupyter and launch again 
export PATH="/home/ec2-user/anaconda2/bin:$PATH"

## Install gcc 
sudo yum install gcc

## Install Jupyter notebook
sudo yum install python-pip
sudo -H pip install jupyter

## Define password hash
Pass: this is my password
Sha1: sha1:18dfaae5d0f8:cdd4683f2918e8482311e24adc22166d29489204

## Install cert pem 
jupyter notebook --generate-config
mkdir certs
cd certs
sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem

## Edit jupyter notebook configuration file 
vim ~/.jupyter/jupyter_notebook_config.py

c = get_config()
c.IPKernelApp.pylab = 'inline' 
c.NotebookApp.certfile = u'/home/ec2-user/certs/mycert.pem' 
c.NotebookApp.ip = '*' 
c.NotebookApp.open_browser = False 

## Your password below  
c.NotebookApp.password = u'sha1:18dfaae5d0f8:cdd4683f2918e8482311e24adc22166d29489204' 
c.NotebookApp.port = 8888

## Run jupyter notebook in background  
jupyter notebook & 

## Go to browser and accept the connection and insert password (not sha1)
https://[IP]:8888/tree/

# Upload ipynb and dataset and you’re all set!
