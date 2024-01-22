""" Find all parallelepipeds that are an optimal solution to the taxicab distance problem, and group them by uniqueness. """
from itertools import product, combinations
from multiprocessing import Pool, cpu_count
from math import acos, degrees
import json
import numpy as np

n = 8  # Define the tessellation range
sequence = [0] + [j for i in range(1, n + 1) for j in (i, -i)]

def is_valid_solution(matrix):
    """
    Dot the matrix with each vector in [0, 1, -1, 2, -2, ..., n, -n]^3 and
    check if the taxicab distance from that point to the origin violates the constraint.
    """
    for vector in product(sequence, repeat=3):
        # Tesselate the parallelepiped by multiplying the matrix by the vector
        coords = np.dot(matrix.T, np.array(vector))
        distance = np.sum(np.abs(coords))  # Calculate the taxicab distance from the origin
        if distance in [1, 2, 3, 4]:
            return False
    return True


def generate_vectors():
    """Generate all vectors with a taxicab distance of 5 from the origin"""
    vectors = []

    for coords in product(range(-5, 5), repeat=3):
        vector = np.array(coords)
        if sum(abs(vector)) == 5:
            vectors.append(vector)
    return vectors


def parallel_search_worker(vectors_subset, vectors):
    """
    Find valid solutions and minimum volume for a given set of vectors.

    Args:
        vectors_subset (list): Subset of vectors to consider.
        vectors (list): All vectors.

    Returns:
        tuple: A tuple containing the list of valid solutions and the minimum volume.
    """
    solutions = np.array([])
    min_volume = np.inf
    vectors_subset = np.array(vectors_subset)
    vectors = np.array(vectors)

    for a in vectors_subset:
        # Filter out 'b' vectors that are too close to 'a'
        distances = np.sum(np.abs(vectors - a), axis=1)
        valid_b_vectors = vectors[distances >= 5]

        # Filter out 'b' vectors where cross product is zero (not linearly independent)
        independent_b_vectors = valid_b_vectors[np.any(np.cross(a, valid_b_vectors) != 0, axis=1)]

        for b, c in combinations(independent_b_vectors, 2):
            cross_prod = np.cross(b, c)
            if not np.any(cross_prod):  # Check if 'b' and 'c' are linearly independent
                continue

            # Create a 3x3 matrix with 'a', 'b', and 'c' as rows
            matrix = np.array([a, b, c], dtype=int)

            # Calculate the volume of the parallelepiped using the determinant
            volume = abs(int(np.linalg.det(matrix)))

            if 0 < volume <= min_volume:
                if is_valid_solution(matrix):
                    if volume < min_volume:
                        min_volume = volume
                        solutions = np.array([matrix])
                    elif volume == min_volume:
                        solutions = np.append(solutions, [matrix], axis=0)

    return solutions, min_volume

def calculate_angles(matrix:np.ndarray):
    """
    Calculate the angles between three vectors in a given matrix.

    Parameters:
    matrix (np.ndarray): A 3x3 matrix representing three vectors as rows.

    Returns:
    tuple: A tuple of three angles (in degrees) between the vectors, sorted in ascending order.
    """
    # Calculate the dot products between the vectors
    dot_ab, dot_ac, dot_bc = map(lambda x: np.dot(x[0], x[1]), combinations(matrix, 2))

    # Calculate the magnitudes of the vectors
    mag_a, mag_b, mag_c = map(np.linalg.norm, matrix)

    # Calculate the cosine of the angles between the vectors
    cos_angle1 = dot_ab / (mag_a * mag_b)
    cos_angle2 = dot_ac / (mag_a * mag_c)
    cos_angle3 = dot_bc / (mag_b * mag_c)

    # Convert to angles (in degrees)
    angle1, angle2, angle3 = map(lambda x: degrees(acos(x)), [cos_angle1, cos_angle2, cos_angle3])

    # Sort angles for consistency and return
    return tuple(sorted([angle1, angle2, angle3]))

def group_by_angle_sets(solutions:list):
    """ Group solutions by the angles between the vectors. """
    angle_groups = {}
    for matrix in solutions:
        angles = calculate_angles(matrix)
        if angles in angle_groups:
            angle_groups[angles].append(matrix)
        else:
            angle_groups[angles] = [matrix]
    return angle_groups

def parallel_search(vectors, positive_vectors):
    """
    Find valid solutions and minimum volume for a given set of vectors.

    Args:
        vectors (list): All vectors.
        positive_vectors (list): All vectors with positive coordinates.

    Returns:
        tuple: A tuple containing the list of valid solutions and the minimum volume.
    """
    # Split positive_vectors into chunks for multiprocessing
    chunks = np.array_split(positive_vectors, cpu_count())

    # Create a multiprocessing pool
    with Pool() as pool:
        # Map the worker function over the chunks
        results = pool.starmap(parallel_search_worker, [(chunk, vectors) for chunk in chunks])

    # Combine results from all workers
    solutions = []
    min_volume = np.inf
    for sol, vol in results:
        if vol < min_volume:
            min_volume = vol
            solutions = sol
        elif vol == min_volume:
            solutions = np.append(solutions, sol, axis=0)
    return solutions, min_volume

def main():
    """ Main function. """
    vectors = generate_vectors()

    # We can arbitrarily make one vector positive
    positive_vectors = [v for v in vectors if v[0] >= 0 and v[1] >= 0 and v[2] >= 0]

    solutions, min_volume = parallel_search(vectors, positive_vectors)

    angle_groups = group_by_angle_sets(solutions)

    # Save solutions to .csv as integers
    np.savetxt("solutions.csv", solutions.reshape(-1, 9), delimiter=",", fmt="%d")
    print("Solutions saved to solutions.csv")

    angle_groups_prepared = {}
    for angles, matrices in angle_groups.items():
        angle_groups_prepared[str(angles)] = [matrix.tolist() for matrix in matrices]

    with open("angle_groups.json", "w", encoding="utf-8") as f:
        json.dump(angle_groups_prepared, f, indent=4, default=lambda x: x.tolist())
    print("Angle groups saved to angle_groups.txt")
    # Transpose solution matrices to get vectors as rows
    print("Optimal Vectors:")
    for matrix in solutions:
        print(matrix)
    print("Minimum Volume:", min_volume)


if __name__ == "__main__":
    main()
