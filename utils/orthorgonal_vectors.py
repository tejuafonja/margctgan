import numpy as np

def random_orthogonal_vector(vectors):
    # Ensure that vectors is a 2D NumPy array
    vectors = np.array(vectors)
    
    # Check if vectors are linearly dependent
    if np.linalg.matrix_rank(vectors) < len(vectors):
        raise ValueError("Input vectors are linearly dependent. Cannot find orthogonal vector.")
    
    # Generate a random vector
    random_vector = np.random.rand(vectors.shape[1])
    
    # Orthogonalize the random vector with respect to the given vectors
    for vector in vectors:
        random_vector -= (np.dot(random_vector, vector) / np.dot(vector, vector)) * vector
    
    # Normalize the orthogonal vector to have unit length
    random_vector /= np.linalg.norm(random_vector)
    
    return random_vector



def random_orthogonal_matrix(n, k):
  given_vectors = np.array([np.random.rand(n)])

  norm_given_vectors = np.linalg.norm(given_vectors)
  given_vectors = given_vectors / norm_given_vectors

  for i in range(k-1):
    orthogonal_vector = random_orthogonal_vector(given_vectors)
    given_vectors = np.vstack([given_vectors, orthogonal_vector])  
  
  return given_vectors.T


if __name__ == "__main__":
  given_vectors = np.array([[1, 0, 0], [0, 1, 0]])
  orthogonal_vector = random_orthogonal_vector(given_vectors)
  assert (np.array([0, 0, 1]) == orthogonal_vector).all()

  k=10
  n=100
  out = random_orthogonal_matrix(n=n, k=k)
  assert (np.isclose(out.T @ out, 0) + np.eye(k)).all()