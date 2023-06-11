#import "thesis.typ": thesis
#import "substitutions.typ"

#show: doc => thesis(
  doc,
)
#show: substitutions.general

#include "introduction.typ"
#include "theory.typ"
#include "related_work.typ"
#include "approach.typ"
#include "evaluation.typ"
#include "results.typ"
#include "discussion.typ"
#include "conclusion.typ"

#bibliography("bibliography.yml")
