#import "thesis.typ": thesis
#import "substitutions.typ"

#show footnote.entry: substitutions.subs

#show: doc => thesis(
  doc,
)

#substitutions.subs[
  #outline()

  #include "introduction.typ"
  #include "theory.typ"
  #include "related_work.typ"
  #include "approach.typ"
  #include "evaluation.typ"
  #include "results.typ"
  #include "discussion.typ"
  #include "conclusion.typ"
]

#bibliography("bibliography.yml")
