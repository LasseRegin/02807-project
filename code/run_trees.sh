
# Parameters
#export MAX_POSTS=1000000


# Extract posts
echo "Extracting posts.."
python3 preprocess.py

# Transform posts
echo "Transforming posts.."
python3 transform.py

# Train
echo "Training model.."
python3 train_trees_para.py

# Evaluate
echo "Evaluating model.."
python3 eval_trees.py

echo "Done!"
