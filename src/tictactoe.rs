/// Tic Tac Toe game interface.
use std::fmt;

use crate::board_game;

use board_game::Cell::{Empty, Full};
use board_game::EndState::{Draw, Winner};
use board_game::GameState::{Ended, Ongoing};
use board_game::Player::{P1, P2};
use board_game::{
    Cell, GameBoard, GameMove, GameState, Player, ALPHABET, CELL_TAKEN, NO_MOVE_TO_UNDO,
    OUT_OF_RANGE,
};

/// Store the size and state of the tic-tac-toe board.
#[derive(Clone, PartialEq)]
pub struct TicTacToeBoard {
    // dimension of the board (total number of cells = size * size)
    pub size: usize,
    // for example: 3x3 grid would be a vec of length 9
    pub cells: Vec<Cell>,
    // the history of the moves played: (P,(R,C)) means player P made a move at row R, col C
    pub move_history: Vec<TicTacToeMove>,
}

impl fmt::Debug for TicTacToeBoard {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
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
            formatter,
            "TicTacToeBoard {{\n  size: {},\n  cells: [{}],\n  history: {:?}\n}}",
            self.size, board_repr, self.move_history,
        )
    }
}

/// The information needed to enter a move to the tic tac toe board.
/// (row, col, player)
pub type TicTacToeMove = (usize, usize, Player);

impl GameMove for TicTacToeMove {
    fn player(&self) -> Player {
        self.2
    }
    fn set_player(&mut self, p: Player) {
        self.2 = p;
    }
}

impl GameBoard<TicTacToeMove> for TicTacToeBoard {
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
                        P1 => 'o',
                        P2 => 'x',
                    },
                });
            }
            println!("{}", row);
        }
        println!("\n");
    }

    /// Does P1 get to make the next move?
    fn is_p1_turn(&self) -> bool {
        match self.move_history.last() {
            None | Some(&(_, _, P2)) => true,
            Some(&(_, _, P1)) => false,
        }
    }

    /// Used in the theoretical playouts that tree search agents use to backtrack after reaching an
    /// end state. Enables more efficient search b/c we don't need to create copies of the board.
    fn undo_move(&mut self) -> Result<(), &str> {
        match self.move_history.pop() {
            Some((r, c, _)) => {
                self.cells[r * self.size + c] = Empty;
                Ok(())
            }
            None => Err(NO_MOVE_TO_UNDO),
        }
    }

    /// Return a vector of (row, col) legal moves the player can choose.
    #[allow(clippy::match_bool)]
    fn get_valid_moves(&self) -> Vec<TicTacToeMove> {
        let mut valid_moves = Vec::new();
        for (i, cell) in self.cells.iter().enumerate() {
            if let Empty = cell {
                let player = match self.is_p1_turn() {
                    true => P1,
                    false => P2,
                };
                valid_moves.push((i / self.size, i % self.size, player));
            }
        }
        valid_moves
    }

    /// Return Ok(Ended(_) if game is over, Ok(Ongoing) if it continues, Err if invalid move.
    fn enter_move(&mut self, move_: TicTacToeMove) -> Result<GameState, &str> {
        let (row, col, player) = move_;

        if row >= self.size || col >= self.size {
            return Err(OUT_OF_RANGE);
        }
        let idx = row * self.size + col;
        match &self.cells[idx] {
            Empty => self.cells[idx] = Full(player),
            _ => return Err(CELL_TAKEN),
        }
        self.move_history.push((row, col, player));

        if self.move_wins_game(row, col, player) {
            return Ok(Ended(Winner(player)));
        }

        if self.get_valid_moves().is_empty() {
            return Ok(Ended(Draw));
        }

        Ok(Ongoing)
    }

    fn move_history(&self) -> Vec<TicTacToeMove> {
        self.move_history.clone()
    }
}

/// Representation of an N-dimensional tic-tac-toe board.
impl TicTacToeBoard {
    /// Return a new Board of (size * size) cells.
    pub fn new(size: usize) -> TicTacToeBoard {
        TicTacToeBoard {
            cells: vec![Empty; (size * size) as usize],
            size,
            move_history: vec![],
        }
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
    let size = 3;
    // row
    let mut board = TicTacToeBoard::new(size);
    for i in 0..size {
        board.cells[i * board.size] = Full(P1);
    }
    assert!(board.player_fills_line(P1, &|i| i % board.size == 0));

    // col
    let mut board = TicTacToeBoard::new(size);
    for i in 0..size {
        board.cells[i] = Full(P1);
    }
    assert!(board.player_fills_line(P1, &|i| i / board.size == 0));

    // diag \
    let mut board = TicTacToeBoard::new(size);
    for i in 0..size {
        board.cells[i * board.size + i] = Full(P1);
    }
    assert!(board.player_fills_line(P1, &|i| i % board.size == i / board.size));

    // diag /
    let mut board = TicTacToeBoard::new(size);
    for i in 0..size {
        board.cells[board.size + i * board.size - i - 1] = Full(P1);
    }
    assert!(
        board.player_fills_line(P1, &|i| (i % board.size) + (i / board.size)
            == board.size - 1)
    );
}

#[test]
fn test_enter_move() {
    let size = 3;
    let mut board = TicTacToeBoard::new(size);
    assert!(board.is_p1_turn());
    assert!(board.get_valid_moves().len() == size * size);

    assert!(Ok(Ongoing) == board.enter_move((1, 1, P1)));
    assert!(!board.is_p1_turn());
    assert!(board.get_valid_moves().len() == size * size - 1);

    assert!(Ok(Ongoing) == board.enter_move((1, 2, P2)));
    assert!(board.is_p1_turn());
    assert!(board.get_valid_moves().len() == size * size - 2);
    assert!(Err(CELL_TAKEN) == board.enter_move((1, 2, P1)));
    assert!(Err(OUT_OF_RANGE) == board.enter_move((1, 3, P1)));
}

#[test]
fn test_undo_move() {
    let size = 3;
    let mut board = TicTacToeBoard::new(size);
    // draw game
    assert!(Ok(Ongoing) == board.enter_move((1, 1, P1)));
    assert!(Ok(Ongoing) == board.enter_move((1, 2, P2)));
    assert!(Ok(Ongoing) == board.enter_move((1, 0, P1)));
    assert!(Ok(Ongoing) == board.enter_move((0, 2, P2)));
    assert!(Ok(Ongoing) == board.enter_move((2, 2, P1)));
    assert!(Ok(Ongoing) == board.enter_move((0, 1, P1)));
    assert!(Ok(Ongoing) == board.enter_move((0, 0, P2)));
    assert!(Ok(Ongoing) == board.enter_move((2, 0, P1)));
    assert!(Ok(Ended(Draw)) == board.enter_move((2, 1, P2)));
    assert!(board.move_history.len() == 9);
    // undo last two moves and make P1 win
    assert!(Ok(()) == board.undo_move());
    assert!(!board.is_p1_turn());
    assert!(Ok(()) == board.undo_move());
    assert!(board.is_p1_turn());
    assert!(Ok(Ended(Winner(P1))) == board.enter_move((2, 1, P1)));
    for _ in 0..8 {
        assert!(Ok(()) == board.undo_move());
    }
    assert!(board.move_history.is_empty());
    assert!(Err(NO_MOVE_TO_UNDO) == board.undo_move());
}
