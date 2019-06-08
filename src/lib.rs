/// Tic-tac-toe implementation and a Monte Carlo tree search agent.
extern crate rand;
#[macro_use]
extern crate cached;

use std::io;
use std::ops::Add;
use std::sync::mpsc;
use std::thread;

use rand::{thread_rng, Rng};

use Cell::{Empty, Full};
use EndState::{Draw, Winner};
use GameState::{Ended, Ongoing};
use Player::{P1, P2};

// number of random games to play out from a given game state
// stop after we reach or exceed this number
// on my Ryzen 2600 w/threading, it takes about 5 seconds to generate this many playouts
const PLAYOUT_BUDGET: usize = 1_000_000;

const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

// Error messages
const BAD_INPUT: &str = "bad input";
const OUT_OF_RANGE: &str = "out of range";
const CELL_TAKEN: &str = "cell taken";

cached! {
    FACTORIAL;
    fn factorial(i: usize) -> usize = {
        if i <= 1 {
            return i;
        }

        factorial(i - 1) * i
    }
}

/// At a given game state, the summed wins/losses/draw scores, as well as the total number of
/// playouts that have been tried.
#[derive(Debug, PartialEq)]
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

/// Did the game end in a draw or was there a winner?
pub enum EndState {
    Winner(Player),
    Draw,
}

/// Has the game ended or is it ongoing?
pub enum GameState {
    Ended(EndState),
    Ongoing,
}

/// Used for deciding whose turn it is
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    fn get_opponent(&self) -> Player {
        match self {
            P1 => P2,
            P2 => P1,
        }
    }
}

/// An agent that can choose a move from a tic-tac-toe board.
pub trait TicTacToeAgent {
    fn choose_move(&self, board: &Board) -> (usize, usize);
}

/// An agent controlled by the user running the program.
pub struct HumanAgent {}

impl TicTacToeAgent for HumanAgent {
    fn choose_move(&self, _board: &Board) -> (usize, usize) {
        loop {
            println!("Enter a move (like \"a0\"):");
            match self.get_move() {
                Ok(rc) => return rc,
                Err(_) => {
                    println!("Oops, enter valid input");
                }
            };
        }
    }
}

impl HumanAgent {
    pub fn new() -> HumanAgent {
        HumanAgent {}
    }

    /// Accept player input from stdin, parse into (row, col) indexes.
    /// Columns are letter indexes, rows are integers.
    /// Example: "a2" means column 0, row 2
    fn get_move(&self) -> Result<(usize, usize), &'static str> {
        let mut input = String::new();
        if let Err(_) = io::stdin().read_line(&mut input) {
            return Err(BAD_INPUT);
        }
        // at least 2 indices + \n
        if input.len() < 3 {
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

        Ok((row, col))
    }
}

#[derive(Clone, Debug)]
/// AI agent that plays using Monte Carlo tree search to choose moves.
/// TODO eventually store interior node scores here so we don't need to check entire tree of
/// possible games at every turn
pub struct MonteCarloAgent {
    player: Player,
}

impl TicTacToeAgent for MonteCarloAgent {
    /// Agent chooses the best available move
    fn choose_move(&self, board: &Board) -> (usize, usize) {
        println!("{:?} AI is thinking...", self.player);
        let valid_moves = self.get_valid_moves(board);
        let num_moves = valid_moves.len();

        let mut max_score = -((2 as isize).pow(62));
        let mut total_playouts = 0;
        let mut best_rowcol = valid_moves[0];

        let (sender, receiver) = mpsc::channel();

        for (row, col) in valid_moves {
            // need a mutable copy here so we can use recursive backtracking without needing to make
            // a copy of the board at each step
            let mut theoretical_board = board.clone();
            let theoretical_self = self.clone();
            let theoretical_player = self.player.clone();
            let new_sender = sender.clone();
            thread::spawn(move || {
                let outcomes = theoretical_self.score_move(
                    &mut theoretical_board,
                    theoretical_player,
                    row,
                    col,
                    // our "playout budget" for a single move is the total budget split evenly
                    // between all the current possible moves
                    PLAYOUT_BUDGET / num_moves,
                );
                new_sender.send((outcomes, row, col)).unwrap();
            });
        }

        let mut threads_finished = 0;
        for (outcomes, row, col) in receiver {
            threads_finished += 1;

            println!("Evaluating move (row {}, col {}), {:?}", row, col, outcomes);

            if outcomes.score > max_score {
                best_rowcol = (row, col);
                max_score = outcomes.score;
            }

            total_playouts += outcomes.total;

            if threads_finished == num_moves {
                break;
            }
        }

        println!(
            "Chosen move: {:?}, total playouts: {}",
            best_rowcol, total_playouts
        );

        best_rowcol
    }
}

impl MonteCarloAgent {
    pub fn new(player: Player) -> MonteCarloAgent {
        MonteCarloAgent { player }
    }

    fn get_valid_moves(&self, board: &Board) -> Vec<(usize, usize)> {
        let mut valid_moves = Vec::new();
        for (i, cell) in board.cells.iter().enumerate() {
            if let Empty = cell {
                valid_moves.push((i / board.size, i % board.size));
            }
        }
        valid_moves
    }

    /// Scores a given move by playing it out on a theoretical board alternating between the agent and
    /// the opponent player taking turns until it reaches an end state as many times as it can
    /// before it reaches its playout_threshold.
    fn score_move(
        &self,
        board: &mut Board,
        player: Player,
        row: usize,
        col: usize,
        playout_budget: usize,
    ) -> Outcomes {
        // play the move in question on the theoretical board
        if let Ok(Ended(endstate)) = board.enter_move(row, col, player) {
            // backtrack once we're done calculating
            board.undo_move(row, col);

            // score is factorial of the number of empty cells remaining because that's how many
            // different end states there could be if we were able to keep playing moves after
            // someone wins the game, up until the board is full - this is a way of weighting
            // more immediate (less moves to get to) wins/losses more heavily than farther out ones
            let score = factorial(board.num_cells_remaining()) + 1;

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
        let mut valid_moves = self.get_valid_moves(board);
        thread_rng().shuffle(&mut valid_moves);
        let opp = player.get_opponent();

        // recurse to the possible subsequent moves and score them
        let mut outcomes = Outcomes::new(0, 0);
        for (new_r, new_c) in &valid_moves {
            outcomes = outcomes + self.score_move(board, opp, *new_r, *new_c, playout_budget);

            if outcomes.total >= playout_budget {
                // we've met or surpassed the total # of games we're supposed to play out
                break;
            }
        }

        // backtrack once we're done calculating
        board.undo_move(row, col);

        outcomes
    }
}

/// Represents a single cell of the tic-tac-toe board.
#[derive(Clone, Copy, Debug)]
pub enum Cell {
    Empty,
    Full(Player),
}

/// Store the size and state of the tic-tac-toe board.
#[derive(Clone, Debug)]
pub struct Board {
    // dimension of the board (total number of cells = size * size)
    size: usize,
    // for example: 3x3 grid would be a vec of length 9
    cells: Vec<Cell>,
    // who gets to make the next move?
    pub is_p1_turn: bool,
}

/// Representation of an N-dimensional tic-tac-toe board
impl Board {
    /// Return a new Board of (size * size) cells.
    pub fn new(size: usize) -> Board {
        Board {
            cells: vec![Empty; (size * size) as usize],
            size,
            is_p1_turn: true,
        }
    }

    /// Print the board to stdout with the column and row labels:
    ///
    ///   abc
    /// 0 ...
    /// 1 ...
    /// 2 ...
    ///
    /// NOTE: this will fail with self.size > 10, need to implement smarter indexing
    pub fn display(&self) {
        if self.size > 10 {
            println!("The board is too large to print with indices");
            return;
        }
        println!("\n  {}", &ALPHABET.to_string()[..self.size]);
        for i in 0..self.size {
            let mut row = String::with_capacity(self.size as usize);
            row.push_str(&format!("{:<2}", i));
            for j in 0..self.size {
                row.push(match &self.cells[i * self.size + j] {
                    Empty => '.',
                    Full(player) => match player {
                        P1 => 'o',
                        P2 => 'x',
                    },
                });
            }
            println!("{}", row);
        }
        println!("\n");
    }

    fn undo_move(&mut self, r: usize, c: usize) {
        self.cells[r * self.size + c] = Empty;
    }

    fn num_cells_remaining(&self) -> usize {
        let empties: Vec<&Cell> = self
            .cells
            .iter()
            .filter(|c| match c {
                Empty => true,
                _ => false,
            })
            .collect();

        empties.len()
    }

    fn no_moves_remaining(&self) -> bool {
        self.num_cells_remaining() == 0
    }

    /// Return Ok(Some(player)) if game is over, Ok(None) if it continues, Err if invalid move.
    pub fn enter_move(
        &mut self,
        row: usize,
        col: usize,
        player: Player,
    ) -> Result<GameState, &str> {
        let idx = row * self.size + col;
        if idx > self.size * self.size - 1 {
            return Err(OUT_OF_RANGE);
        }
        match &self.cells[idx] {
            Empty => self.cells[idx] = Full(player),
            _ => return Err(CELL_TAKEN),
        }

        if self.move_wins_game(row, col, player) {
            return Ok(Ended(Winner(player)));
        }

        if self.no_moves_remaining() {
            return Ok(Ended(Draw));
        }

        self.is_p1_turn = !self.is_p1_turn;

        Ok(Ongoing)
    }

    /// Return if the line defined by the filter_fn is filled with cells of type player.
    fn line_is_filled_with_player(&self, player: Player, filter_fn: &Fn(usize) -> bool) -> bool {
        let matching_cells: Vec<(usize, &Cell)> = self
            .cells
            .iter()
            .enumerate()
            .filter(|&(i, x)| {
                filter_fn(i)
                    && match x {
                        Full(cplayer) => &player == cplayer,
                        _ => false,
                    }
            })
            .collect();

        matching_cells.len() == self.size
    }

    /// Did the last move played at (r,c) by player win the game?
    fn move_wins_game(&self, r: usize, c: usize, player: Player) -> bool {
        // check row
        if self.line_is_filled_with_player(player, &|i| i / self.size == r) {
            return true;
        }

        // check col
        if self.line_is_filled_with_player(player, &|i| i % self.size == c) {
            return true;
        }

        // check \ diagonal if relevant
        if self.line_is_filled_with_player(player, &|i| i % self.size == i / self.size) {
            return true;
        }

        // check / diagonal if relevant
        if self.line_is_filled_with_player(player, &|i| {
            (i % self.size) + (i / self.size) == self.size - 1
        }) {
            return true;
        }

        false
    }
}
