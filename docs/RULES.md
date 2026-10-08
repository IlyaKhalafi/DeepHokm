# Hokm rules as implemented

Every rule variant this project pins, so behaviour is unambiguous and
testable. The engine is the single authority: the web UI and the training
environment both sit on top of it rather than reimplementing anything.

Hokm is a four-player partnership trick-taking game. Seats 0-3 play clockwise;
teams A = {0, 2} and B = {1, 3} sit opposite each other. The implementation pins
these deliberate design decisions (no other variants are supported):

- Standard 52-card deck; suits clubs, diamonds, hearts, spades (ids 0-3); ranks
  2..10, J, Q, K, A with 2 lowest and A highest; card id = 13 * suit + rank.
- The first **hakem** (trump caller) is a uniformly random seat. The hakem is
  dealt 5 cards and declares the trump suit, then all 52 cards are dealt so
  every player holds exactly 13 (the hakem 5 + 8).
- The hakem leads the first trick. Players must follow the led suit when able,
  otherwise may play any card. The highest trump wins the trick; absent trump,
  the highest card of the led suit wins. The winner leads next. A hand runs up
  to 13 tricks but ends the moment either team captures 7. The remaining cards
  are not played.
- Capturing 7 tricks wins the hand for the team and scores 1 game point.
  No kot/bustom bonus in v1.
- First team to 7 game points wins the match. One RL episode = one full match.
  If the hakem's team won the hand the hakem stays; otherwise the hakem passes
  to seat `(old_hakem + 1) % 4`.
- Each player observes only public information plus their own hand.
