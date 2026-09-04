"""Verhoeff checksum validation for Indian Aadhaar identifiers.

The Verhoeff algorithm is a checksum formula for error detection with decimal
numbers, based on the dihedral group D5. UIDAI specifies Verhoeff for 12-digit
Aadhaar numbers.
"""

D_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

P_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

INV_TABLE = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(number: str) -> bool:
    """Validates whether a numeric string satisfies the Verhoeff checksum."""
    clean_digits = [int(c) for c in reversed(number) if c.isdigit()]
    if not clean_digits or len(clean_digits) < 2:
        return False

    checksum = 0
    for i, digit in enumerate(clean_digits):
        checksum = D_TABLE[checksum][P_TABLE[i % 8][digit]]
    return checksum == 0


def generate_verhoeff_check_digit(number: str) -> int:
    """Generates the single-digit Verhoeff checksum to append to a number."""
    clean_digits = [int(c) for c in reversed(number) if c.isdigit()]
    checksum = 0
    for i, digit in enumerate(clean_digits):
        checksum = D_TABLE[checksum][P_TABLE[(i + 1) % 8][digit]]
    return INV_TABLE[checksum]
