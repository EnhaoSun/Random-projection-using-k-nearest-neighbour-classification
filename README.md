# Random projection using k-nearest neighbour classification

## Problem Description
Given a set of labeled examples, a k-nearest neighbour classifier predicts the label of a test data point x in the following way: It first finds the k-nearest neighbours of x in the dataset using suitable distance metric, and predicts the label to be the majority of the labels among k-nearest neighbours; ties are broken arbitrarily. The nearest neighbours of a test data point x are obtained by simply computing the Euclidean distances from x to all points in the dataset. [More about KNN]( http://www.inf.ed.ac.uk/teaching/courses/iaml/2011/slides/knn.pdf)

## Data set
News group dataset. Popular 20 newsgroup [dataset](http://qwone.com/~jason/20Newsgroups/) contains posts on different topics. Each post is represented as a bag-of-words — a vector indexed by words; The value at index i indicates the number of times i-th word occurs in that post. There are two types of files:
* The **dataset_vector.csv** file is a sparse representation of the bag-of-words. It has three fields: postid, wordid, wordcount; e.g., the line "6 312 1" says that 6-th post has the 312-th word once. Note, wordid takes a value between 1 and 61200.
* **A dataset_label.csv** file contains the label (topic it belongs) of post i at line number i.
The **test_vector.csv** and **test_label.csv** files have the same format as the corresponding dataset files. In this exercise, use **dataset_vector.csv** and **dataset_label.csv** to predict the labels of the examples in **test_vector.csv** and finally evaluate the prediction using **test_label.csv**.

## Dimension reduction using random projection.
To reduce a d-dimensional vector v to a m- dimensional vector v, create a matrix M with dimension (m,d), where each M(i)(j) is an i.i.d. from a normal distribution with mean 0 and variance 1 and then compute v = M · v. Here M · v represents a standard matrix-vector product.

## Limit on external libraries
Won't use numpy and scipy.

## Problem 1: (Answer in /source/q1)
For each pair of vectors u,v in **dataset_vector.csv** and their reduced dimensional representations u ̃, v ̃ with m = 100, compute the ratio of Euclidean distances, i.e., ∥u ̃−v ̃∥2 / ∥u−v∥2

a) Plot a histogram of these ratios. While plotting ignore the point pairs with near zero distances. x-axis of the histogram should have the ratio values, and y-axis should denote how many times that appeared — this is standard histogram.

## Problem 2: (Answer in /source/q2)
Build a k-nearest neighbour classifier as described above. Reduce the dimensionality of vectors in both **dataset_vector.csv** and **test_vector.csv** using random projection with m = 100. Run the classifier using these reduced dimensional representations, i.e., predict the classes for the m-dimensional vectors using the k-nearest neighbours found among m-dimensional dataset vectors. Use the **evaluate.py** to get the prediction error from predicted_label.csv; Prediction error is the percentage of misclassified test instances. On the dataset provided, with k = 1 it should not take longer than 3.5 seconds to execute and should have prediction error at most 30%. With increasing k = 10, 20 the execution time should remain almost the same but the prediction error should decrease to 25% and 22% respectively.

## Code
source - contains all the source files

run.sh - script to compile and run the program with the following options: 
* ./run.sh compile : Compiles the source. (optional)
* ./run.sh q1 <m> <dataset vector>: Shows the plot asked in Problem 1(a).
* ./run.sh q2 <k> <m> <dataset vector> <dataset label> <test vector> <test label> :
k-nearest neighbour classification with reduced dimensions as asked in Problem 2.
