# 1. Import the tools we need
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 2. Load the data (Measurements of 150 flowers)
iris = load_iris()
X, y = iris.data, iris.target

# 3. Split the data: 80% for learning, 20% for a "final exam"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Pick a model (K-Nearest Neighbors: it looks at similar examples to guess)
model = KNeighborsClassifier(n_neighbors=3)

# 5. THE TRAINING: The model finds patterns in the data
model.fit(X_train, y_train)

# 6. TEST IT: How accurate is our AI?
accuracy = model.score(X_test, y_test)
print(f"Our AI is {accuracy * 100}% accurate!")
