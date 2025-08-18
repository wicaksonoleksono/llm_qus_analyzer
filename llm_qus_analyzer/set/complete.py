# Are there missing prerequisite stories for a task to be completed?
# Does the set enable a fully functional application?
# Are there no critical missing steps?
# Implementing a set of user stories creates a feature-complete application, no steps are missingsd

# need to parse user story witht he same object .
# All user story , pairwise , and shit like that .
# added user story this is very very important
_description = """
**Evaluate whether the user story set is 'Complete' based on [Means] action dependencies:**
**Scope: Cross-story [Means] coverage analysis**

1. **[Means] Action Analysis:**
   - Extract **action verb** and **direct object** from each story's [Means]
   - Identify if action verb **requires prerequisite actions** on same direct object
   - Common dependencies: view→create, edit→create, delete→create, submit→create, pause→play, checkout→add

2. **Prerequisite Coverage Check:**
   - For each story with **dependent action**, search entire story set for **prerequisite stories**
   - Must find stories covering **all required prerequisite actions** on **same direct object**
   - Check: Does any story provide the **foundational action** needed?

3. **Gap Detection:**
   - Identify **missing prerequisite stories** that would prevent functional implementation
   - Report **void dependencies** where action references non-existent objects/states
   - Flag **workflow breaks** where logical sequence is incomplete

**Evaluation Questions:**
- Can every [Means] action be **logically performed** given the existing story set?
- Are there **foundational stories missing** that other stories depend on?
- Would implementing these stories create a **functionally complete** application?

**Incorrect:** Action verbs referencing objects/states that no story creates or establishes
**Expected:** All prerequisite actions covered, enabling complete functional workflows

**Example Dependencies:**
- edit/view/delete → create
- submit/send → create/draft  
- pause/stop → start/play
- checkout/remove → add
- approve/reject → submit
- configure/modify → install/setup
"""
