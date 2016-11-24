
# Extract posts
echo "Extracting posts.."
python3 preprocess.py

# Transform posts
echo "Transforming posts.."
python3 transform.py

# Train
echo "Training model.."
python3 train_para.py

# Evaluate
echo "Evaluating model.."
python3 eval.py

echo "Done!"
