/// Monte Carlo tree search Tic-Tac-Toe agent and command line interface for playing against it.
extern crate rand;

use std::io;
use std::ops::Add;

use rand::{thread_rng, Rng};

use Cell::{Empty, Full};
use EndState::{Draw, Winner};
use GameState::{Ended, Ongoing};
use Player::{Human, AI};

// TODO add command line flags to control these
const STARTING_PLAYER: Player = Human;
const BOARD_SIZE: usize = 3;
// number of random games to play out from a given game state
// stop after we reach or exceed this number
const PLAYOUTS_THRESHOLD: usize = 100_000;

const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

// error messages
const BAD_INPUT: &str = "bad input";
const OUT_OF_RANGE: &str = "out of range";
const CELL_TAKEN: &str = "cell taken";

/// At a given game state, the summed wins/losses/draws
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

enum EndState {
    Winner(Player),
    Draw,
}

enum GameState {
    Ended(EndState),
    Ongoing,
}

/// The type of player
#[derive(Clone, Copy, Debug, PartialEq)]
enum Player {
    // You
    Human,
    // MonteCarloAgent (or other AI agent)
    AI,
}

impl Player {
    /// Get the opponent (opposite enum) of this player
    fn get_opponent(self) -> Player {
        match self {
            Human => AI,
            AI => Human,
        }
    }
}

/// Represents a single cell of the tic-tac-toe board
#[derive(Clone, Copy, Debug)]
enum Cell {
    Empty,
    Full(Player),
}

/// Store the size and state of the tic-tac-toe board
#[derive(Clone, Debug)]
struct Board {
    // dimension of the board (total number of cells = size * size)
    size: usize,
    // for example: 3x3 grid would be a vec of length 9
    cells: Vec<Cell>,
}

fn factorial(i: usize) -> usize {
    if i <= 1 {
        return i;
    }
    factorial(i - 1) * i
}

// TODO eventually store interior node scores here so we don't need to check entire tree of
// possible games at every turn
struct MonteCarloAgent {}

impl MonteCarloAgent {
    fn new() -> MonteCarloAgent {
        MonteCarloAgent {}
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

    /// Scores a given move by playing it out recursively on a theoretical board alternating
    /// between the AI and Human players taking turns until it reaches an end state
    fn score_move(
        &self,
        board: &mut Board,
        player: Player,
        r: usize,
        c: usize,
        playout_threshold: usize,
    ) -> Outcomes {
        // play the move in question on the theoretical board
        if let Ok(Ended(endstate)) = board.enter_move(r, c, player) {
            // backtrack once we're done calculating
            board.undo_move(r, c);

            // score is factorial of the number of empty cells remaining because that's how many
            // different end states there could be if we were able to keep playing moves after
            // someone wins the game, up until the board is full - this is a way of weighting
            // more immediate (less moves to get to) wins/losses more heavily than farther out ones
            let score = factorial(board.num_cells_remaining()) + 1;

            // return score if win, -score if lose, 0 if draw
            return match endstate {
                Winner(AI) => Outcomes::new(score as isize, 1),
                Winner(Human) => Outcomes::new(-(score as isize), 1),
                Draw => Outcomes::new(0, 1),
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
            outcomes = outcomes + self.score_move(board, opp, *new_r, *new_c, playout_threshold);

            if outcomes.total >= playout_threshold {
                // we've met or surpassed the total # of games we're supposed to play out
                break;
            }
        }

        // backtrack once we're done calculating
        board.undo_move(r, c);

        outcomes
    }

    /// Agent chooses the best available move
    /// TODO: parallelize using threads
    fn choose_move(&self, board: &Board) -> (usize, usize) {
        // get valid moves in random order
        let valid_moves = self.get_valid_moves(board);
        let num_moves = valid_moves.len();

        let mut max_score = -((2 as isize).pow(62));
        let mut total_playouts = 0;
        let mut best_rc = valid_moves[0];
        // need a mutable copy here so we can use recursive backtracking without needing to make
        // a copy of the board at each step
        let mut theoretical_board = board.clone();

        for (r, c) in valid_moves {
            let outcomes = self.score_move(
                &mut theoretical_board,
                AI,
                r,
                c,
                PLAYOUTS_THRESHOLD / num_moves,
            );

            println!("Evaluating move (r: {}, c: {}), {:?}", r, c, outcomes);

            if outcomes.score > max_score {
                best_rc = (r, c);
                max_score = outcomes.score;
            }

            total_playouts += outcomes.total;
        }

        println!(
            "Chosen move: {:?}, total playouts: {}",
            best_rc, total_playouts
        );

        best_rc
    }
}

impl Board {
    /// Return a new Board of (size * size) cells
    fn new(size: usize) -> Board {
        Board {
            cells: vec![Empty; (size * size) as usize],
            size,
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
    fn display(&self) {
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
                        Human => 'o',
                        AI => 'x',
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

    /// Return Ok(Some(player)) if game is over, Ok(None) if it continues, Err if invalid move
    fn enter_move(&mut self, row: usize, col: usize, player: Player) -> Result<GameState, &str> {
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

        Ok(Ongoing)
    }

    /// Return if the line defined by the filter_fn is filled with cells of type player
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

/// Accept player input from stdin, parse into (row, col) indexes
/// Columns are letter indexes, rows are integers
/// Example: "a2" means column 0, row 2
fn get_move() -> Result<(usize, usize), &'static str> {
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

fn main() -> io::Result<()> {
    let mut board = Board::new(BOARD_SIZE);
    let mut player = STARTING_PLAYER;
    let ai = MonteCarloAgent::new();

    println!("\nIT'S TIC-TAC-TOEEEEEEE TIIIIIIIIME!!!!!!");
    board.display();

    loop {
        let rc: (usize, usize);
        match player {
            Human => {
                println!("Enter a move (like \"a0\"):");

                rc = match get_move() {
                    Ok(rc) => rc,
                    Err(_) => {
                        println!("Oops, enter valid input");
                        continue;
                    }
                };
            }
            AI => {
                println!("AI is thinking...");
                rc = ai.choose_move(&board);
            }
        }

        let (row, col) = rc;
        match board.enter_move(row, col, player) {
            Ok(Ended(endstate)) => {
                board.display();
                match endstate {
                    Winner(Human) => println!("Game over, you won!"),
                    Winner(AI) => println!("Game over, you lost!"),
                    Draw => println!("Game over, it's a draw!"),
                }
                return Ok(());
            }
            Ok(Ongoing) => board.display(),
            Err(msg) => {
                // if the AI entered an illegal move, something is broken
                assert!(player == Human);
                println!("Oops, illegal move: {}", msg);
                // back to start of loop to let user enter a different move
                continue;
            }
        };
        player = player.get_opponent();
    }
}
