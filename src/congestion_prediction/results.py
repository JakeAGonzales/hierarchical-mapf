import numpy as np

def analyze_matrices(file_path="results_.npz"):
    # Load the results
    print("Loading results from:", file_path)
    data = np.load(file_path)
    
    # Extract matrices
    A = data['A']
    L = data['L']
    X = data['X']
    B = data['B']
    
    # Function to print stats for a matrix
    def print_matrix_stats(name, matrix):
        print(f"\n{name} Matrix Statistics:")
        print(f"Shape: {matrix.shape}")
        print(f"Mean: {np.mean(matrix):.4f}")
        print(f"Max:  {np.max(matrix):.4f}")
        print(f"Min:  {np.min(matrix):.4f}")
        print(f"Std:  {np.std(matrix):.4f}")
        print(f"Non-zero elements: {np.count_nonzero(matrix)}")
        
    # Print stats for each matrix
    print_matrix_stats("A", A)
    print_matrix_stats("L", L)
    print_matrix_stats("X", X)
    print_matrix_stats("B", B)
    
    # Print a few sample values from each matrix
    num_samples = 5
    print(f"\nFirst {num_samples} values from each matrix:")
    print("\nA[:5]:")
    print(A[:5])
    print("\nL[:5]:")
    print(L[:5])
    print("\nX[:5]:")
    print(X[:5])
    print("\nB[:5]:")
    print(B[:5])

if __name__ == "__main__":
    analyze_matrices()