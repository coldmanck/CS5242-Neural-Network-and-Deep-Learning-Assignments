All instructions for CS5242 Assignment 3 on RNN is in ‘CS5242_Assignment_3.pdf’. For Python users, you can use the IPython notebook and the helper functions to code. We strongly recommend you to use the IPython notebook as it provides means to check your code and gives helper function to put your code together. For non-Python users, you can use the pdf and accompanying python function APIs to know what is expected of the submission.

ASSIGNMENT 3 DEADLINE: 24 NOV 2017 (FRI) 11.59PM

<<Python virtual environment setup>>

For Python users, we recommend using virtual environment to set up the python3 and the relevant dependencies. The instructions here should suffice, but more details can be found at: http://cs231n.github.io/assignments2017/assignment3/

$ cd assignment3
$ sudo pip install virtualenv
$ virtualenv -p python3 .env       # Create a virtual environment (python3)
$ source .env/bin/activate         # Activate the virtual environment
$ pip install -r requirements.txt  # Install dependencies

# do your work…

$ deactivate 		# to exit the virtual environment

Each time you want to work on the assignment, you have to activate the environment.

For this assignment, the dataset has already been downloaded and the word dictionary has already been generated. The dataset we use here comes from a kaggle competition, see the full dataset on https://www.kaggle.com/c/si650winter11/data. We only sample 1000 sentences from the original training set as our dataset. train.csv and test.csv are generated with the ratio of 4:1 from the 1000 samples.

<<Starting IPython>>
For those unfamiliar with IPython, this tutorial gives a quick introduction:
http://cs231n.github.io/ipython-tutorial/

$ cd assignment
$ jupyter notebook
# then, open up CS5242_Assignment_3.ipynb


