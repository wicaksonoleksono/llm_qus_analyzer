from dataclasses import dataclass


@dataclass
class Violation:
    """Represents a quality violation found in a user story analysis.

    This class captures information about a specific quality issue, including
    which components of the user story are affected and suggestions for improvement.

    Example:
    >>> violation = Violation(
    ...     parts={'role'},
    ...     issue='The role is missing',
    ...     suggestion='Add a specific role like "customer" or "admin"'
    ... )
    >>> 'role' in violation.parts
    True
    """

    parts: set[str]
    """
    Set of user story components related to the violation.
    Typical values: 'role', 'means', 'ends', or combinations.
    """

    issue: str
    """Description of the specific quality issue found."""

    suggestion: str
    """Recommended action to fix the violation."""
