#!/usr/bin/env python3
"""
A function def minor(matrix): that calculates the minor matrix of a matrix
"""


def minor(matrix):
    """
    a function that calculates the minor matrix of a matrix.
    """
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]] or not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    minorMatrix = []
    for i in range(len(matrix)):
        minorRow = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in
                          (matrix[:i] + matrix[i+1:])]
            minorRow.append(determinant(sub_matrix))
        minorMatrix.append(minorRow)
    return minorMatrix


def determinant(matrix):
    """
    Calculating the determinant of a matrix and returning it.
    """
    if matrix == [[]]:
        return 1

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    n = len(matrix)
    det = 0
    for j in range(n):
        sign = (-1)**j
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += sign * matrix[0][j] * determinant(sub_matrix)
    return det
