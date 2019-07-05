/// Tic Tac Toe game interface.
use std::fmt;

use Cell::{Empty, Full};
use EndState::{Draw, Winner};
use GameState::{Ended, Ongoing};
use Player::{P1, P2};

pub const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

const OUT_OF_RANGE: &str = "out of range";
const CELL_TAKEN: &str = "cell taken";

/// Did the game end in a draw or was there a winner?
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EndState {
    Winner(Player),
    Draw,
}

/// Has the game ended or is it ongoing?
#[derive(Clone, Copy, Debug, PartialEq)]
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
    pub fn get_opponent(self) -> Player {
        match self {
            P1 => P2,
            P2 => P1,
        }
    }
}

/// An agent that can choose a move from a tic-tac-toe board. Self is mutable because AI agents
/// may need to update their state as they search for moves.
pub trait TicTacToeAgent {
    fn choose_move(&mut self, board: &TicTacToeBoard) -> (usize, usize);
}

/// Represents a single cell of the tic-tac-toe board.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cell {
    Empty,
    Full(Player),
}

/// Store the size and state of the tic-tac-toe board.
#[derive(Clone, PartialEq)]
pub struct TicTacToeBoard {
    // dimension of the board (total number of cells = size * size)
    pub size: usize,
    // for example: 3x3 grid would be a vec of length 9
    pub cells: Vec<Cell>,
    // who gets to make the next move?
    pub is_p1_turn: bool,
}

impl fmt::Debug for TicTacToeBoard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut board_repr = String::new();
        for cell in self.cells.iter() {
            let cell_repr = match cell {
                Empty => '.',
                Full(P1) => 'o',
                Full(P2) => 'x',
            };
            board_repr.push(cell_repr);
        }
        write!(
            f,
            "{{ TicTacToeBoard size: {}, cells: [{}], is_p1_turn: {} }}",
            self.size, board_repr, self.is_p1_turn
        )
    }
}

// TODO make generic GameBoard trait so agents can be used with other board games
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
    pub fn undo_move(&mut self, r: usize, c: usize) {
        self.cells[r * self.size + c] = Empty;
    }

    /// Return a vector of (row, col) legal moves the player can choose.
    pub fn get_valid_moves(&self) -> Vec<(usize, usize)> {
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

        if self.get_valid_moves().is_empty() {
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
        if row == col && self.player_fills_line(player, &|i| i % self.size == i / self.size) {
            return true;
        }

        // check / diag
        if row + col == self.size - 1
            && self.player_fills_line(player, &|i| {
                (i % self.size) + (i / self.size) == self.size - 1
            })
        {
            return true;
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
