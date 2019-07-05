### _NOTE: under construction_

# Monte Carlo tree search Tic-Tac-Toe AI

An AI agent that can play Tic-Tac-Toe using [Monte
Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) tree search, and an
interface for playing against it, written in Rust.

```
$ cargo run

IT'S TIC-TAC-TOEEEEEEE TIIIIIIIIME!!!!!!

  abc
0 ...
1 ...
2 ...


Enter a move (like "a0"):
b1

  abc
0 ...
1 .o.
2 ...


P2 MCTSAgent is thinking...

  abc
0 x..
1 .o.
2 ...
```

## Features:
- Reports estimated score, number of playouts, and time spent for move evaluations
- Can use the interface to play against the AI, have it play against itself, or play
  against another human player (change `agent1` and `agent2` in `main.rs`)
- Can tweak the strength of the AI by setting the maximum amount of simulated playouts per
  turn (change `PLAYOUT_BUDGET` constant)
- Multiple agents
  - __MCTSAgent__
    - Stores an internal tree of possible game states
    - Retains states from previous moves
    - Based on [pbsinclair42/MCTS](https://github.com/pbsinclair42/MCTS)
  - __ForgetfulSearchAgent__ (not a Monte Carlo search agent)
    - With a playout budget (number of theoretical games for the AI to play
      until a win/loss/draw per turn) of 1,000,000, it plays normal `BOARD_SIZE = 3`
      Tic-Tac-Toe (as well as 4) optimally, usually does fairly well at 5
      - With normal Tic-Tac-Toe, the number of possible games is 9! or 362,880, which is
        less than this budget, so it is able to evaluate every possible game state
        at any move
    - Multithreaded 
      - Can compute roughly 6-7 million playouts/sec on a 5x5 board (maximum game move depth
        of 25) with a six-core Ryzen 2600 and `--release` build
    - Only has purely random tree search for move evaluation
  - __RandomAgent__
    - Just chooses random (legal) moves
  - __HumanAgent__
    - Lets the user choose moves through a CLI user interface

## TODO:
- Use [UCT](https://link.springer.com/chapter/10.1007%2F11871842_29) to choose nodes to
  expand 
- Detect symmetrical game states
- Tree pruning heuristics
  - Example: explore moves that block the opponent from getting (board size - 1) pieces
    in a single row/column/diagonal first
- Let AI choose its own heuristics
  - Train it against itself
  - Look for patterns in the winning AI's play
    - Example: board is in a certain generalizable, non-exact state and the winning AI
      usually chose a certain move or pattern of moves
  - During move evaluation, play out these good moves before random ones
- Generalize the agent to more than just Tic-Tac-Toe
  - Requires domain-specific functions:
    - Get valid moves from a given game state
    - Score the end state of the game
    - Optionally, a heuristic function that will tell the agent which moves it should
      explore first
  - Other games to target:
    - Chess
    - [Ty Overby](https://github.com/TyOverby)'s "Four-not-three" hex grid game
    - Something not 2D-grid based?
- Match recordings
  - Write during a match to a file, replay in UI through a pair of `RecordedAgent`s
- Command line flags for changing board size, player types, playout budget parameters
- Tests (maybe this should be higher in the list)
