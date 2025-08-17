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


@dataclass
class PairwiseViolation:
    """Represents a set violation found in user story set analysis.
    This class captures information about specific quality issues between
    two user stories, including which components are affected and suggestions
    for improvement.

    Example:
    >>> violation = PairwiseViolation(
    ...     first_parts={'means'},
    ...     second_parts={'means'},
    ...     issue='The second story refers to the same means but has contradicting ends',
    ...     suggestion='Stories can be concatenated into one by using more general action
terms'
    ... )
    >>> 'means' in violation.first_parts and 'means' in violation.second_parts
    True
    """

    first_parts: set[str]
    """Components affected in the first user story."""

    second_parts: set[str]
    """Components affected in the second user story."""

    issue: str
    """Description of the specific quality issue between the stories."""

    suggestion: str
    """Recommended action to fix the violation."""


@dataclass
class FullSetViolation:
    """Represents a violation found across multiple user stories in a set.

    This class captures quality issues that involve multiple stories,
    with their IDs, affected components, and suggestions for resolution.

    Example:
    >>> violation = FullSetViolation(
    ...     story_ids=[1, 3, 7],
    ...     parts_per_story=[{'means'}, {'ends'}, {'means', 'ends'}],
    ...     issue='Stories 1, 3, and 7 have conflicting requirements for the same feature',
    ...     suggestion='Consolidate these stories or clarify their different scopes'
    ... )
    >>> violation.story_ids
    [1, 3, 7]
    """
    story_ids: list[int]
    """List of story indices/IDs that are involved in this violation."""

    parts_per_story: list[set[str]]
    """Components affected for each story (same order as story_ids)."""

    issue: str
    """Description of the quality issue across the story set."""

    suggestion: str
    """Recommended action to fix the violation."""

    def get_story_parts(self, story_id: int) -> set[str]:
        """Get the affected parts for a specific story ID."""
        try:
            idx = self.story_ids.index(story_id)
            return self.parts_per_story[idx]
        except ValueError:
            return set()

    @property
    def affected_stories_count(self) -> int:
        """Number of stories involved in this violation."""
        return len(self.story_ids)
