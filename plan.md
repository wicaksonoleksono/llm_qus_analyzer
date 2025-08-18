In set extraction there are 5 criteria
conflict free: Meaning a user story should not be inconsistent with another user story

Unique : Every user story is unique, Duplicates are avoided
Uniform : Has the same template
Independent: A user story is self-contained and has no inhereit depedencies on other stories
Complete: Implementing a set of user story create a feature complete application,
no steps are missng.

Where we can get this.
isFullDuplicate us1 == us2
semanticDuplication syn(us1 != us2) \and sem(us1 == us2)
sameMeansDifferentEnds M1 == M2 \and E1 != E2
differentRoleSameStory R1 !=R2 \and m1==m2 or E1 \int E2 atleast 1 element
CasualitiedhasDep(u1,u2) -> dependes(au1,au2)
ObjectDepedencies(us1,us2) -> \forevery ou1 \subset ou2

conflict-free, different means same ends,
unique Semantic duplication or Syntactic duplication
uniform ,Tempalte checking.
independent,object depedency
complete, Full featured story.

cs v
po v
un v
cf v
fs
es
unique
uniform
independent
complete
