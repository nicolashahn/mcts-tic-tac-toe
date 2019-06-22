/// Monte Carlo tree search agent for Tic-tac-toe.
extern crate rand;

use std::io;
use std::ops::Add;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use Cell::{Empty, Full};
use EndState::{Draw, Winner};
use GameState::{Ended, Ongoing};
use Player::{P1, P2};

// TODO add command line flags to control board size, player agent types, playout budget
pub const BOARD_SIZE: usize = 4;
// number of random games to play out from a given game state
// stop after we reach or exceed this number
const PLAYOUT_BUDGET: usize = 1_000_000;

const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

// Error messages
const BAD_INPUT: &str = "bad input";
const OUT_OF_RANGE: &str = "out of range";
const CELL_TAKEN: &str = "cell taken";

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
    fn choose_move(&self, board: &TicTacToeBoard) -> (usize, usize);
}

/// An agent controlled by the user running the program.
pub struct HumanAgent {}

impl TicTacToeAgent for HumanAgent {
    fn choose_move(&self, _board: &TicTacToeBoard) -> (usize, usize) {
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
/// AI agent that plays using tree search to choose moves - but does not remember tree expansions
/// that have been calculated in prior moves. Not a true Monte Carlo tree search agent.
pub struct ForgetfulSearchAgent {
    player: Player,
}

impl TicTacToeAgent for ForgetfulSearchAgent {
    /// Agent chooses the best available move
    fn choose_move(&self, board: &TicTacToeBoard) -> (usize, usize) {
        println!("{:?} AI is thinking...", self.player);
        let valid_moves = board.get_valid_moves();
        let num_moves = valid_moves.len();

        let mut max_score = -((2 as isize).pow(62));
        let mut total_playouts = 0;
        let mut best_rowcol = valid_moves[0];

        let (sender, receiver) = mpsc::channel();

        let now = Instant::now();
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
            "Chosen move: {:?}, total playouts: {}, choosing took {:?}",
            best_rowcol,
            total_playouts,
            now.elapsed()
        );

        best_rowcol
    }
}

impl ForgetfulSearchAgent {
    pub fn new(player: Player) -> ForgetfulSearchAgent {
        ForgetfulSearchAgent { player }
    }

    /// Scores a given move by playing it out on a theoretical board alternating between the agent and
    /// the opponent player taking turns until it reaches an end state as many times as it can
    /// before it reaches its playout_threshold.
    fn score_move(
        &self,
        board: &mut TicTacToeBoard,
        player: Player,
        row: usize,
        col: usize,
        playout_budget: usize,
    ) -> Outcomes {
        // play the move in question on the theoretical board
        if let Ok(Ended(endstate)) = board.enter_move(row, col, player) {
            // backtrack once we're done calculating
            board.undo_move(row, col);

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
pub struct TicTacToeBoard {
    // dimension of the board (total number of cells = size * size)
    size: usize,
    // for example: 3x3 grid would be a vec of length 9
    cells: Vec<Cell>,
    // who gets to make the next move?
    pub is_p1_turn: bool,
}

/// Representation of an N-dimensional tic-tac-toe board
impl TicTacToeBoard {
    /// Return a new Board of (size * size) cells.
    pub fn new(size: usize) -> TicTacToeBoard {
        TicTacToeBoard {
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

    /// Used in the theoretical playouts that tree search agents use to backtrack after reaching an
    /// end state. Enables more efficient search b/c we don't need to create copies of the board.
    fn undo_move(&mut self, r: usize, c: usize) {
        self.cells[r * self.size + c] = Empty;
    }

    /// Return a vector of (row, col) legal moves the player can choose.
    fn get_valid_moves(&self) -> Vec<(usize, usize)> {
        let mut valid_moves = Vec::new();
        for (i, cell) in self.cells.iter().enumerate() {
            if let Empty = cell {
                valid_moves.push((i / self.size, i % self.size));
            }
        }
        valid_moves
    }

    /// Return Ok(Ended(_) if game is over, Ok(Ongoing) if it continues, Err if invalid move.
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

        if self.get_valid_moves().len() == 0 {
            return Ok(Ended(Draw));
        }

        self.is_p1_turn = !self.is_p1_turn;

        Ok(Ongoing)
    }

    /// Return if the line defined by the filter_fn is filled with cells of type player.
    fn player_fills_line(&self, player: Player, filter_fn: &Fn(usize) -> bool) -> bool {
        let mut player_count = 0;
        for i in 0..self.size * self.size {
            if filter_fn(i) {
                if let Full(p) = self.cells[i] {
                    if p == player {
                        player_count += 1;
                    }
                }
            }
        }

        player_count == self.size
    }

    /// Did the last move played at (row, col) by player win the game?
    fn move_wins_game(&self, row: usize, col: usize, player: Player) -> bool {
        // check row
        if self.player_fills_line(player, &|i| i / self.size == row) {
            return true;
        }

        // check col
        if self.player_fills_line(player, &|i| i % self.size == col) {
            return true;
        }

        // check \ diag
        if row == col {
            if self.player_fills_line(player, &|i| i % self.size == i / self.size) {
                return true;
            }
        }

        // check / diag
        if row + col == self.size - 1 {
            if self.player_fills_line(player, &|i| {
                (i % self.size) + (i / self.size) == self.size - 1
            }) {
                return true;
            }
        }

        false
    }
}

#[test]
fn test_player_fills_line() {
    // row
    let mut board = TicTacToeBoard::new(3);
    for i in 0..3 {
        board.cells[i * board.size] = Full(P1);
    }
    assert!(board.player_fills_line(P1, &|i| i % board.size == 0));

    // col
    let mut board = TicTacToeBoard::new(3);
    for i in 0..3 {
        board.cells[i] = Full(P1);
    }
    assert!(board.player_fills_line(P1, &|i| i / board.size == 0));

    // diag \
    let mut board = TicTacToeBoard::new(3);
    for i in 0..3 {
        board.cells[i * board.size + i] = Full(P1);
    }
    assert!(board.player_fills_line(P1, &|i| i % board.size == i / board.size));

    // diag /
    let mut board = TicTacToeBoard::new(3);
    for i in 0..3 {
        board.cells[board.size + i * board.size - i - 1] = Full(P1);
    }
    assert!(
        board.player_fills_line(P1, &|i| (i % board.size) + (i / board.size)
            == board.size - 1)
    );
}
