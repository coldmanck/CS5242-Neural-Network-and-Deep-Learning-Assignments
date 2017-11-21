All instructions for CS5242 Assignment 2 is in ‘CS5242_Assignment_2.pdf’. For Python users, you can use the IPython notebook and the helper functions to code. We strongly recommend you to use the IPython notebook as it provides means to check your code and gives helper function to put your code together. For non-Python users, you can use the pdf and accompanying python function APIs to know what is expected of the submission.

ASSIGNMENT 2 DEADLINE: 19 OCT 2017 (THU) 11.59PM

<<Python virtual environment setup>>

For Python users, we recommend using virtual environment to set up the python3 and the relevant dependencies. The instructions here should suffice, but more details can be found at: http://cs231n.github.io/assignments2017/assignment2/

$ cd assignment2
$ sudo pip install virtualenv
$              # Create a virtual environment (python3)
$ source .env/bin/activate         # Activate the virtual environment
$ pip install -r requirements.txt # Install dependencies

# do your work…

$ deactivate 		# to exit the virtual environment

Each time you want to work on the assignment, you have to activate the environment.


<<Get CIFAR10 data>>
We will be using CIFAR10 for this assignment. For Python users, simply run:
$cd code_base/datasets
$./get_datasets.sh

For non-Python users, download the data from the official website:
https://www.cs.toronto.edu/~kriz/cifar.html

<<Starting IPython>>
For those unfamiliar with IPython, this tutorial gives a quick introduction:
http://cs231n.github.io/ipython-tutorial/

$ cd assignment
$ jupyter notebook
# then, open up CS5242_Assignment_2.ipynb


