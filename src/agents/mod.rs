/// Agents for TicTacToe.
mod mcts_agent;
pub use mcts_agent::MCTSAgent;

use std::f64;
use std::io;
use std::ops::Add;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::board_game;

use board_game::EndState::{Draw, Winner};
use board_game::GameState::Ended;
use board_game::Player::{P1, P2};
use board_game::{GameBoard, GameMove, Player, RCPMove, ALPHABET};

/*
 * -----------
 * Human Agent
 * -----------
 */

const BAD_INPUT: &str = "bad input";

/// An agent that will choose a valid move given the state of the game board.
pub trait BoardGameAgent<GM, GB>
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    fn choose_move(&mut self, board: &GB) -> GM;
}

/// An agent controlled by the user running the program.
pub struct HumanAgent {
    pub player: Player,
}

impl<GB> BoardGameAgent<RCPMove, GB> for HumanAgent
where
    GB: GameBoard<RCPMove>,
{
    fn choose_move(&mut self, _board: &GB) -> RCPMove {
        loop {
            println!("Enter a move (like \"a0\"):");
            match self.get_user_input() {
                Ok(move_) => {
                    return move_;
                }
                Err(_) => {
                    println!("Oops, enter valid input");
                }
            };
        }
    }
}

impl HumanAgent {
    pub fn new(player: Player) -> HumanAgent {
        HumanAgent { player }
    }
}

/// Trait that translates user input into a GameMove.
pub trait GetsUserInput<GM>
where
    GM: GameMove,
{
    /// Receive input from the user and translate it into a GameMove.
    fn get_user_input(&self) -> Result<GM, &'static str>;
}

/// Currently, the only move type for HumanAgents implemented is RCPMove.
impl GetsUserInput<RCPMove> for HumanAgent {
    /// Accept player input from stdin, parse into (row, col) indexes.
    /// Columns are letter indexes, rows are integers.
    /// Example: "a2" means column 0, row 2
    /// TODO allow multi-digit row input to boost maximum grid size from 10 to 26
    fn get_user_input(&self) -> Result<RCPMove, &'static str> {
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return Err(BAD_INPUT);
        }
        // col letter, row number, '\n'
        if input.len() != 3 {
            return Err(BAD_INPUT);
        }
        let player_move = input.trim();

        let col = match ALPHABET.find(player_move.chars().nth(0).unwrap()) {
            Some(idx) => idx,
            None => return Err(BAD_INPUT),
        };

        let row = match player_move.chars().nth(1).unwrap().to_digit(10) {
            Some(idx) => idx as usize,
            None => return Err(BAD_INPUT),
        };

        Ok((row, col, self.player))
    }
}

/*
 * ------------
 * Random Agent
 * ------------
 */

#[derive(Clone, Debug)]
/// Agent that makes random moves.
pub struct RandomAgent {
    player: Player,
}

impl<GM, GB> BoardGameAgent<GM, GB> for RandomAgent
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    fn choose_move(&mut self, board: &GB) -> GM {
        Self::get_random_move_choice(board)
    }
}

impl RandomAgent {
    pub fn new(player: Player) -> RandomAgent {
        RandomAgent { player }
    }

    pub fn get_random_move_choice<GM: GameMove>(board: &impl GameBoard<GM>) -> GM {
        let valid_moves = board.get_valid_moves();
        let mut rng = thread_rng();
        *valid_moves.choose(&mut rng).unwrap()
    }
}

/*
 * ----------------------
 * Forgetful Search Agent
 * ----------------------
 */

/// At a given game state, the summed wins/losses/draw scores, as well as the total number of
/// playouts that have been tried.
#[derive(Clone, Debug, PartialEq)]
struct Outcomes {
    score: isize,
    total: usize,
}

impl Outcomes {
    fn new(score: isize, total: usize) -> Outcomes {
        Outcomes { score, total }
    }
}

impl Add for Outcomes {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            score: self.score + other.score,
            total: self.total + other.total,
        }
    }
}

#[derive(Clone, Debug)]
/// AI agent that plays using tree search to choose moves - but does not remember tree expansions
/// that have been calculated in prior moves. Not a true Monte Carlo tree search agent, BUT it's
/// blazing fast at raw playout computation.
pub struct ForgetfulSearchAgent {
    player: Player,
    playout_budget: usize,
}

impl<GM, GB> BoardGameAgent<GM, GB> for ForgetfulSearchAgent
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    fn choose_move(&mut self, board: &GB) -> GM {
        let theoretical_board = board.clone();
        self._choose_move(&theoretical_board)
    }
}

impl ForgetfulSearchAgent {
    pub fn new(player: Player, playout_budget: usize) -> ForgetfulSearchAgent {
        ForgetfulSearchAgent {
            player,
            playout_budget,
        }
    }

    /// Agent chooses the best available move
    #[allow(clippy::explicit_counter_loop)]
    fn _choose_move<GM: GameMove>(&mut self, board: &(impl GameBoard<GM>)) -> GM {
        println!("{:?} ForgetfulSearchAgent is thinking...", self.player);
        let valid_moves = board.get_valid_moves();
        let num_moves = valid_moves.len();

        let mut max_score = -((2 as isize).pow(62));
        let mut total_playouts = 0;
        let mut best_move = valid_moves[0];

        let (sender, receiver) = mpsc::channel();

        let now = Instant::now();
        for move_ in valid_moves {
            // need a mutable copy here so we can use recursive backtracking without needing to make
            // a copy of the board at each step
            let mut theoretical_board = board.clone();
            let theoretical_self = self.clone();
            let new_sender = sender.clone();
            // our "playout budget" for a single move is the total budget split evenly
            // between all the current possible moves
            let move_budget = self.playout_budget / num_moves;
            thread::spawn(move || {
                let outcomes =
                    theoretical_self.score_move(&mut theoretical_board, move_, move_budget);
                new_sender.send((outcomes, move_)).unwrap();
            });
        }

        let mut threads_finished = 0;
        for (outcomes, move_) in receiver {
            threads_finished += 1;

            println!("Evaluating move {:?}: {:?}", move_, outcomes);

            if outcomes.score > max_score {
                best_move = move_;
                max_score = outcomes.score;
            }

            total_playouts += outcomes.total;

            if threads_finished == num_moves {
                break;
            }
        }

        println!(
            "
Chosen move:      {:?}
Total playouts:   {}
Choosing took:    {:?}
Playout rate:     {:.2}/sec",
            best_move,
            total_playouts,
            now.elapsed(),
            (total_playouts as f64 / (now.elapsed().as_nanos() as f64)) * 1_000_000_000.0
        );

        best_move
    }

    /// Scores a given move by playing it out on a theoretical board alternating between the agent and
    /// the opponent player taking turns (by recursively calling itself) until it reaches an end
    /// state as many times as it can before it reaches its playout_threshold.
    fn score_move<GM: GameMove>(
        &self,
        board: &mut impl GameBoard<GM>,
        move_: GM,
        playout_budget: usize,
    ) -> Outcomes {
        // play the move in question on the theoretical board
        if let Ok(Ended(endstate)) = board.enter_move(move_) {
            // backtrack once we're done calculating
            if board.undo_move().is_err() {
                panic!("ForgetfulSearchAgent tried to do an illegal undo_move() at game end, board: {:?}", board);
            }

            // the score (num cells remaining)^2 + 1 weights outcomes that are closer in the search
            // space to the current move higher
            // end states where the board is fuller are less likely
            let cells_remaining = board.get_valid_moves().len();
            let score = cells_remaining * cells_remaining + 1;

            // return score/2 if win, -score if lose, 0 if draw
            // a win is only worth half as much as a loss because:
            //
            //      "To win, first you must not lose."
            //      - Nicolas Hahn
            //
            return match (endstate, self.player) {
                (Winner(P1), P1) | (Winner(P2), P2) => Outcomes::new((score / 2) as isize, 1),
                (Draw, _) => Outcomes::new(0, 1),
                _ => Outcomes::new(-(score as isize), 1),
            };
        }

        // if this is an intermediate node:
        // get next possible moves for the opposing player
        let mut valid_moves = board.get_valid_moves();
        let mut rng = thread_rng();
        valid_moves.shuffle(&mut rng);

        // recurse to the possible subsequent moves and score them
        let mut outcomes = Outcomes::new(0, 0);
        for new_move in valid_moves.iter() {
            outcomes = outcomes + self.score_move(board, *new_move, playout_budget);

            if outcomes.total >= playout_budget {
                // we've met or surpassed the total # of games we're supposed to play out
                break;
            }
        }

        // backtrack once we're done calculating
        if board.undo_move().is_err() {
            panic!("ForgetfulSearchAgent tried to do an illegal undo_move() while unwinding simulation");
        }

        outcomes
    }
}
