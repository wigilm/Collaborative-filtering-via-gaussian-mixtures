# Overview
A mixture model for collaborative filtering is built.  
Given is a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.
## Approach
We assume data is generated from K different Gaussians, so K different types of users. The EM algorithm is used to estimate the mixture from a partially observed rating matrix.

## Project structure

* kmeans → baseline using the K-means algorithm
* naive_em.py → first version of the EM algorithm
* em.py → mixture model for collaborative filtering 
* common.py → common functions for all models 
* main.py → runs the code
* test.py → test implementation of EM for a given test case
* toy_data.txt → a 2D dataset that you will work with in tabs 2-5
* netflix_incomplete.txt → the netflix dataset with missing entries to be completed
* netflix_complete.txt → the netflix dataset with missing entries completed
* test_incomplete.txt → a test dataset to test for you to test your code against our implementation
* test_complete.txt → a test dataset to test for you to test your code against our implementation
* test_solutions.txt → a test dataset to test for you to test your code against our implementation

## Results

Root mean squared error of generated data against actual target values (netflix_complete):  
RMSE: 0.4804908505400684

## Attribution

This project was completed as part of the MITx MicroMasters program in Machine Learning and Data Science. The problem formulation, dataset, and portions of the code structure were provided by the course. All model implementations and analysis were completed independently.

