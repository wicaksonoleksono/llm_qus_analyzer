from typing import Optional


class TemplateParser:
    """A utility class for extracting and templatizing patterns from user stories.

    This class provides methods to identify and replace common patterns (roles, means, ends)
    in user stories with standardized tokens, helping to create templates from concrete examples.
    """

    @classmethod
    def __extract_word_chunk(cls, s: str, include_non_alpha: bool = True) -> list[str]:
        """Splits a string into chunks of alphanumeric and non-alphanumeric characters.

        Args:
            s: The input string to be chunked.
            include_non_alpha: If True, includes non-alphanumeric chunks in the output.

        Returns:
            A list of string chunks where each chunk contains either all alphanumeric
            or all non-alphanumeric characters.

        Example:
            >>> __extract_word_chunk("Hello, world!", True)
            ['hello', ', ', 'world', '!']
        """
        s_low = s.lower()
        ch_status = [ch.isalnum() for ch in s_low]
        s_list: list[str] = []
        sid, eid = 0, 0
        for i in range(1, len(s_low)):
            if ch_status[i] == ch_status[sid]:
                eid = i
            else:
                if include_non_alpha or ch_status[sid]:
                    s_list.append(s_low[sid:eid+1])
                sid, eid = i, i
        s_list.append(s_low[sid:eid+1])

        return s_list

    @classmethod
    def __longest_common_subsequence(cls, list1: list[str], list2: list[str]) -> tuple[int, int]:
        """Finds the longest matching subsequence between two lists of strings.

        Args:
            list1: First list of string chunks to compare.
            list2: Second list of string chunks to compare.

        Returns:
            A tuple (start, end) representing the start and end indices in list1
            where the longest common subsequence with list2 occurs.

        Note:
            Uses dynamic programming to efficiently find the longest match.
            Returns (-1, -1) if no subsequence is found.
        """
        m, n = len(list1), len(list2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        start = [[-1] * (n + 1) for _ in range(m + 1)]
        end = [[-1] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                candidates = [
                    (dp[i - 1][j], start[i - 1][j], end[i - 1][j]),
                    (dp[i][j - 1], start[i][j - 1], end[i][j - 1]),
                ]
                if list1[i - 1] == list2[j - 1]:
                    st = start[i - 1][j - 1]
                    candidates.append((
                        dp[i - 1][j - 1] + 1,
                        st if st >= 0 else i - 1,
                        i - 1
                    ))

                candidates.sort(
                    key=lambda c: c[0] * 10000 + (c[1]-c[2]) * 100 + c[2],
                    reverse=True
                )
                dp[i][j], start[i][j], end[i][j] = candidates[0]

        return start[m][n], end[m][n]

    @classmethod
    def __is_valid_subseq(cls, start: int, end: int, ref_len: int) -> bool:
        """Determines if a found subsequence is valid based on length criteria.

        Args:
            start: Start index of the subsequence.
            end: End index of the subsequence.
            ref_len: Length of the reference sequence being matched against.

        Returns:
            True if the subsequence covers at least 90% of the reference length,
            False otherwise.
        """
        if start < 0 or end < 0:
            return False
        return (end-start)/2+1 >= 0.9 * ref_len

    @classmethod
    def __refine_list(cls, terms: list[str], start: int, end: int, token: str) -> list[str]:
        """Replaces a subsequence in a list with a template token.

        Args:
            terms: The original list of string chunks.
            start: Start index of the subsequence to replace.
            end: End index of the subsequence to replace.
            token: The template token to insert (e.g., '<ROLE>', '<MEANS>').

        Returns:
            A new list with the specified subsequence replaced by the token.
        """
        new_terms: list[str] = []
        done = False
        for i, t in enumerate(terms):
            if i < start or i > end:
                new_terms.append(t)
                continue
            if done:
                continue
            new_terms.append(token)
            done = True
        return new_terms

    @classmethod
    def extract_template(cls, user_story: str, role: list[str], means: Optional[str], ends: Optional[str]) -> tuple[str, Optional[str]]:
        """Extracts a template from a user story by replacing patterns with tokens.

        Args:
            user_story: The complete user story string to process.
            role: List of role terms to identify and replace with <ROLE>.
            means: Means component to identify and replace with <MEANS>.
            ends: Ends component to identify and replace with <ENDS>.

        Returns:
            A tuple containing:
                - The templatized version of the user story
                - Any remaining text that wasn't templatized (or None if fully templatized)

        Note:
            The method processes the user story in this order:
            1. Splits into main part and tail (after first period)
            2. Identifies and replaces means component
            3. Identifies and replaces ends component
            4. Identifies and replaces role component
            5. Handles remaining non-templatized portions with <NON>
        """
        parts = user_story.split('.')
        us = parts[0]
        tail = ''
        if len(parts) > 1:
            tail = '.'.join(parts[1:])

        us_list = cls.__extract_word_chunk(us)
        if means:
            means_list = cls.__extract_word_chunk(means, False)
            start, end = cls.__longest_common_subsequence(us_list, means_list)
            if cls.__is_valid_subseq(start, end, len(means_list)):
                us_list = cls.__refine_list(us_list, start, end, '<MEANS>')
        if ends:
            ends_list = cls.__extract_word_chunk(ends, False)
            start, end = cls.__longest_common_subsequence(us_list, ends_list)
            if cls.__is_valid_subseq(start, end, len(ends_list)):
                us_list = cls.__refine_list(us_list, start, end, '<ENDS>')
        if len(role) > 0:
            role_list = cls.__extract_word_chunk(' '.join(role), False)
            start, end = cls.__longest_common_subsequence(us_list, role_list)
            if cls.__is_valid_subseq(start, end, len(role_list)):
                us_list = cls.__refine_list(us_list, start, end, '<ROLE>')

        terms = us_list[:]

        last_token_idx = -1
        for i in range(len(terms)):
            if terms[i] in ['<ROLE>', '<MEANS>', '<ENDS>']:
                last_token_idx = i

        if last_token_idx < 0:
            tail = user_story
            return '<NON>', tail

        new_terms = terms[:last_token_idx + 1]
        if last_token_idx < len(terms) - 1:
            tail = ''.join(terms[last_token_idx + 1:]) + tail

        tmp = [t for t in tail if t.isalnum()]
        if len(tmp) > 0:
            new_terms.append('<NON>')
        else:
            tail = None
        new_terms = [
            t for t in new_terms
            if len(t) > 0 and (t[0].isalnum() or t in ['<ROLE>', '<MEANS>', '<ENDS>', '<NON>'])
        ]
        return ' '.join(new_terms), tail
