= Approach
<approach>
/*
Hier beschreibst du, was du in der Arbeit neues gemacht hast, und wie du es implementiert
hast.
*/

#import "commands.typ": citet

- MuZero implementation
  - mcts behavior customizeable
    - node selection
    - policy calculation
    - action selection
  - supports arbitrary games
  - pytorch
  - typed codebase, passes mypy
- extensions / variants:
  - efficient zero
  - terminal states
  - chance states
  - multiplayer support: teams, prediction of current player
- interplay of mcts policy and selection function
  - unstable behavior of original setup: UCT scores and visit count policy
- application to carchess

#import "related_work.typ": sscl

=== Symmetric #sscl

As outlined in @rw_effzero, #citet("effzero") already layed the groundwork for
remarkable improvements in sampeff by introducing the #sscl.
However, their adoption of the stop-gradient operation from simsiam~@simsiam may have been
short-sighted:

In simsiam, a Siamese net is used to learn latent images in a self-supervised manner.
The arch consists of two identical subnetworks that take an original and an augmented
input image, respectively, and produce a latrep each.
During training, a distance metric is used to compare these latreps for the same input
image.
By minimizing the distance between these reprs, the net is encouraged to learn a
robust and discriminative latent space that encodes useful features in the images.
However, without careful design, the model can collapse into trivial solutions where all
learned reprs end up being very similar or even identical.
The main idea of simsiam is that a stop-gradient operation effectively solves this
problem.~@simsiam

effzero, on the other hand, operates in an rl setting rather than a self-supervised one.
This means that the environment provides explicit training data in the form of obs
and rews.
This data is used to train the nets in a supervised manner, requiring them to generate
diverse pols, vals, and rews from a given obs and sequence of actions.
The exchange of information between the different nets is facilitated by the latreps.
If the latreps were to collapse, it would become impossible to predict meaningful or
distinct output values.

This insight is consistent with the principle of contrastive self-supervised learning,
where additional negative samples are used to prevent the collapse of latreps:
Distinct input images are fed through the two subnetworks, while the training objective is
to increase the divergence between their corresponding latreps.~@simclr

It has been shown that it is possible to learn significant latreps without the use of
negative samples or the stop-gradient mechanism:
#citet("ss_decorr") demonstrate that introducing a decorrelation in the latent space can
effectively steer the net towards a non-collapsed latrep.
In muzero, demanding different outputs from the pnet and dnet (val/pol and rew,
respectively) for different latreps can be seen as the decorrelation mechanism.

Consequently, there is no risk of latrep collapse in the muzero arch, making the inclusion
of a stop-gradient operation unnecessary.
