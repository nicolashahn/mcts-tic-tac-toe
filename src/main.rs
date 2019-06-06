/// Monte Carlo tree search Tic-Tac-Toe agent and command line interface for playing against it.
use std::io;
use std::ops::Add;

use Cell::{Empty, Full};
use EndState::{Draw, Winner};
use GameState::{Ended, Ongoing};
use Player::{Human, AI};

const STARTING_PLAYER: Player = Human;
const BOARD_SIZE: usize = 3;
const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

// error messages
const BAD_INPUT: &str = "bad input";
const OUT_OF_RANGE: &str = "out of range";
const CELL_TAKEN: &str = "cell taken";

/// At a given game state, the summed wins/losses/draws and total playouts
#[derive(Debug, PartialEq)]
struct Outcomes {
    score: isize,
    total: usize,
}

impl Outcomes {
    fn new(score: isize, total: usize) -> Outcomes {
        Outcomes { score, total }
    }
    fn as_f64(&self) -> f64 {
        self.score as f64 / self.total as f64
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

    /// Scores a given move by playing it out recursively alternating between the AI and Human
    /// players taking turns until it reaches all end states
    fn score_move(&self, board: &mut Board, player: Player, r: usize, c: usize) -> Outcomes {
        // create a new board with the move in question played
        if let Ok(Ended(endstate)) = board.enter_move(r, c, player) {
            // backtrack once we're done calculating
            board.undo_move(r, c);
            // Wins get score of 1, losses -1, draws 0
            // TODO weight outcomes more heavily if they're closer in the tree
            // (less steps to get here from the original choose_move() step)
            return match endstate {
                Winner(AI) => Outcomes::new(1, 1),
                Winner(Human) => Outcomes::new(-1, 1),
                Draw => Outcomes::new(0, 1),
            };
        }

        // if this is an intermediate node:
        // get next possible moves for the opposing player
        let valid_moves = self.get_valid_moves(board);
        let opp = player.get_opponent();

        // recurse to the possible subsequent moves and score them
        let mut total = Outcomes::new(0, 0);
        for (new_r, new_c) in &valid_moves {
            total = total + self.score_move(board, opp, *new_r, *new_c);
        }

        // backtrack once we're done calculating
        board.undo_move(r, c);

        total
    }

    /// Agent chooses the best available move
    fn choose_move(&self, board: &Board) -> (usize, usize) {
        let valid_moves = self.get_valid_moves(board);

        let mut max_score = Outcomes::new(-((2 as isize).pow(62)), 1);
        let mut best_rc = valid_moves[0];
        // need a mutable copy here so we can use recursive backtracking without needing to make
        // a copy of the board at each step
        let mut theoretical_board = board.clone();
        for (r, c) in valid_moves {
            let score = self.score_move(&mut theoretical_board, AI, r, c);
            println!("{} {} {:?}", r, c, score);
            if score.as_f64() > max_score.as_f64() {
                best_rc = (r, c);
                max_score = score;
            }
        }

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

    fn no_moves_remaining(&self) -> bool {
        let empties: Vec<&Cell> = self
            .cells
            .iter()
            .filter(|c| match c {
                Empty => true,
                _ => false,
            })
            .collect();

        empties.len() == 0
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
