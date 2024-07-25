"""This module contains metadata for the project."""

# Define the categories for the digits.
# - A list with all numbers and uppercase letters.
CATEGORIES = [str(x) for x in range(10)] + [chr(x) for x in range(65, 91)]

CLASS_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}

IDX_TO_CLASS = {idx: cat for idx, cat in enumerate(CATEGORIES)}
