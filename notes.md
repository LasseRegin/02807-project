
# Plan

Run the K-means on all posts.


Train extremely randomized trees on segments of the data.
I have the unique words (dictionary) and the unique tags, so it is possible
to train an ensemble of extremely randomized trees (or random forests) all based
on small part of the trainin set. Each tree can be saved to a file, and on
evaluation the trees can be loaded and some kind of majority vote can be used.
