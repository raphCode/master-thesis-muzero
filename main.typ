#import "thesis.typ": thesis
#import "substitutions.typ"

#show: doc => thesis(
  doc,
)
#show: substitutions.general

#include "introduction.typ"
#include "related_work.typ"
#include "approach.typ"

#bibliography("bibliography.yml")
