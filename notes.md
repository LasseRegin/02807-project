
# Idea

Use MapReduce to find N top used tags. Use the tags to find all the posts
with the given tags (maybe also with MapReduce). Use `mrjob`.

Then train some ML model to predict the tags from a given post.


## Very ambitious idea extension

Cluster all posts and train ML models on the posts/tag subset in order to cover
the entire tag space, hence making it possible to predict possible tags for any
given text, and not only the top N tags.




# Runs

Using sequential naive implementation

```
time python3 extraction.py
python3 extraction.py  1507.81s user 77.45s system 97% cpu 27:06.29 total
```
