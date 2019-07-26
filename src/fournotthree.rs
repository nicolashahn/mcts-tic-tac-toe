/// 4not3 ("four!three"?) game interface.
///
/// Rules: The game is played on a hex grid usually of radius 6. Each player alternates laying down
/// a piece anywhere on the board (like TicTacToe). A player wins if they get four or more of their
/// pieces in a row, but loses if exactly three of their pieces are in a row. If a player plays a
/// move that causes both three and four piece rows, the four piece row takes precedence, so that
/// player wins.
use std::fmt;

use crate::board_game;

use board_game::Cell::{Empty, Full, OutOfBounds};
use board_game::EndState::{Draw, Winner};
use board_game::GameState::{Ended, Ongoing};
use board_game::Player::{P1, P2};
use board_game::{
    Cell, GameBoard, GameMove, GameState, Player, RCPMove, ALPHABET, CELL_TAKEN, NO_MOVE_TO_UNDO,
    OUT_OF_BOUNDS, OUT_OF_RANGE,
};

/// Store the size and state of the tic-tac-toe board.
#[derive(Clone, PartialEq)]
pub struct FourNotThreeBoard {
    // Dimension of the board (total number of cells = size * size)
    // Must be an odd number to get a proper hexagon-shaped grid
    pub size: usize,
    // All cells in the hex grid, arranged like so:
    // 0 1 2
    //  3 4 5
    //   6 7 8
    // To achieve the hexagonal board shape, we need to set some cells as filled with neither
    // player (indices 0 and 8 in this example)
    pub cells: Vec<Cell>,
    // Using the row, col in RCPMove as axial coordinates, see
    // https://www.redblobgames.com/grids/hexagons/#coordinates-cube
    pub move_history: Vec<RCPMove>,
}

impl FourNotThreeBoard {
    /// Create a new FourNotThreeBoard with the radius given (number of cells from corner to
    /// center, including both center and corner cells)
    pub fn new(radius: usize) -> FourNotThreeBoard {
        let size = (radius * 2) - 1;
        let mut board = FourNotThreeBoard {
            cells: vec![Empty; (size * size) as usize],
            size,
            move_history: vec![],
        };
        for i in 0..board.cells.len() {
            let (row, col) = (i / size, i % size);
            if row + col < (size / 2) || row + col > (2 * (size - 1)) - (size / 2) {
                board.cells[i] = OutOfBounds;
            }
        }

        board
    }

    /// Given a line of cells, check whether or not a player has either gotten exactly 3 in a row,
    /// meaning the other player won, or 4/5 in a row, meaning they won. Do only one of these
    /// checks, toggled by the `for_win` argument.
    fn check_line(line: &Vec<Cell>, for_win: bool) -> Option<Player> {
        let checker = |ct, cell_type: Cell| match cell_type {
            Full(player) => {
                if for_win && ct > 3 {
                    Some(player)
                } else if !for_win && ct == 3 {
                    Some(player.get_opponent())
                } else {
                    None
                }
            }
            _ => None,
        };
        let mut curr_cell_type: Cell = Empty;
        let mut count = 0;
        for &cell in line {
            if cell == curr_cell_type {
                count += 1;
            } else {
                if let Some(winner) = checker(count, curr_cell_type) {
                    return Some(winner);
                }
                curr_cell_type = cell;
                count = 1;
            }
        }
        if let Some(winner) = checker(count, curr_cell_type) {
            return Some(winner);
        }
        None
    }

    /// Check the game board to see if a player has won, and return who did if so.
    fn maybe_get_winner(&self, move_: RCPMove) -> Option<Player> {
        fn get_filtered_cells(cells: &Vec<Cell>, filter_fn: &Fn(usize) -> bool) -> Vec<Cell> {
            let mut filtered_cells = vec![];
            for i in 0..cells.len() {
                if filter_fn(i) {
                    filtered_cells.push(cells[i]);
                }
            }

            filtered_cells
        }

        let (row, col, _) = move_;
        let move_i = (row * self.size) + col;

        for for_win in &[true, false] {
            let row_cells = get_filtered_cells(&self.cells, &|i| i / self.size == row);
            let col_cells = get_filtered_cells(&self.cells, &|i| i % self.size == col);
            let diag_cells = get_filtered_cells(&self.cells, &|i| {
                (usize::max(i, move_i) - usize::min(i, move_i)) % (self.size - 1) == 0
            });
            for cells in vec![row_cells, col_cells, diag_cells] {
                if let Some(winner) = FourNotThreeBoard::check_line(&cells, *for_win) {
                    return Some(winner);
                }
            }
        }

        None
    }
}
#[test]
fn test_check_line() {
    let ongoing = vec![Empty, Empty, Full(P1), Full(P2), Empty];
    let p1_loss = vec![Empty, Full(P1), Full(P1), Full(P1), Full(P2)];
    let p1_win = vec![Empty, Full(P1), Full(P1), Full(P1), Full(P1)];
    let p1_win_w_5 = vec![Empty, Full(P1), Full(P1), Full(P1), Full(P1), Full(P1)];

    assert_eq!(FourNotThreeBoard::check_line(&ongoing, true), None);
    assert_eq!(FourNotThreeBoard::check_line(&ongoing, false), None);
    assert_eq!(FourNotThreeBoard::check_line(&p1_loss, true), None);
    assert_eq!(FourNotThreeBoard::check_line(&p1_loss, false), Some(P2));
    assert_eq!(FourNotThreeBoard::check_line(&p1_win, true), Some(P1));
    assert_eq!(FourNotThreeBoard::check_line(&p1_win, false), None);
    assert_eq!(FourNotThreeBoard::check_line(&p1_win_w_5, true), Some(P1));
}

#[test]
fn test_maybe_get_winner() {
    let size = 5;
    // size 5 board indices, minus the OutOfBounds
    //       2  3  4
    //     6  7  8  9
    //   10 11 12 13 14
    //    15 16 17 18
    //     20 21 22
    let idx_to_move = |i, p| (i / size, i % size, p);
    let mut board = FourNotThreeBoard::new(size);
    board.cells[22] = Full(P1);
    assert_eq!(board.maybe_get_winner(idx_to_move(22, P1)), None);

    // rows
    for i in 10..13 {
        board.cells[i] = Full(P1);
    }
    assert_eq!(board.maybe_get_winner(idx_to_move(12, P1)), Some(P2));
    board.cells[13] = Full(P1);
    assert_eq!(board.maybe_get_winner(idx_to_move(13, P1)), Some(P1));

    // cols (looks like top-left to bot-right on the displayed board)
    board = FourNotThreeBoard::new(size);
    board.cells[6] = Full(P1);
    board.cells[11] = Full(P1);
    board.cells[16] = Full(P1);
    assert_eq!(board.maybe_get_winner(idx_to_move(16, P1)), Some(P2));
    board.cells[21] = Full(P1);
    assert_eq!(board.maybe_get_winner(idx_to_move(21, P1)), Some(P1));

    // diag (top-right to bot-left)
    board = FourNotThreeBoard::new(size);
    board.cells[9] = Full(P1);
    board.cells[13] = Full(P1);
    board.cells[17] = Full(P1);
    assert_eq!(board.maybe_get_winner(idx_to_move(17, P1)), Some(P2));
    board.cells[21] = Full(P1);
    assert_eq!(board.maybe_get_winner(idx_to_move(21, P1)), Some(P1));
}

impl fmt::Debug for FourNotThreeBoard {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut board_repr = String::new();
        for cell in self.cells.iter() {
            let cell_repr = match cell {
                OutOfBounds => '\\',
                Empty => '.',
                Full(P1) => 'o',
                Full(P2) => 'x',
            };
            board_repr.push(cell_repr);
        }
        write!(
            formatter,
            "FourNotThreeBoard {{\n  size: {},\n  cells: [{}],\n  history: {:?}\n}}",
            self.size, board_repr, self.move_history,
        )
    }
}

impl GameBoard<RCPMove> for FourNotThreeBoard {
    /// Print the board to stdout with the column and row labels:
    ///
    ///   a b c d e
    /// 0      . . . 0
    ///  1    . . . . 1
    ///   2  . . . . . 2
    ///    3  . . . .   3
    ///     4  . . .     4
    ///         a b c d e
    ///
    fn display(&self) {
        if self.size > 26 {
            println!("The board is too large to print with indices");
            return;
        }

        let alpha_w_spaces: String = String::from(&ALPHABET[..self.size])
            .chars()
            .map(|c: char| return format!("{} ", c))
            .collect();
        println!("\n  {}", alpha_w_spaces);

        let mut bottom_label_spacer = String::new();
        for i in 0..self.size {
            let mut row = String::with_capacity(self.size as usize);
            for _ in 0..i {
                row.push(' ');
            }
            row.push_str(&format!("{:<3}", i));
            for j in 0..self.size {
                row.push_str(match &self.cells[i * self.size + j] {
                    OutOfBounds => "  ",
                    Empty => ". ",
                    Full(player) => match player {
                        P1 => "o ",
                        P2 => "x ",
                    },
                });
            }
            println!("{}{}", row, i);
            bottom_label_spacer.push(' ');
        }

        println!("{}   {}\n\n", bottom_label_spacer, alpha_w_spaces);
    }

    /// Does P1 get to make the next move?
    fn is_p1_turn(&self) -> bool {
        match self.move_history.last() {
            Some(&(_, _, P1)) => false,
            _ => true,
        }
    }

    /// Used in the theoretical playouts that tree search agents use to backtrack after reaching an
    /// end state. Enables more efficient search because we don't need to create copies of the
    /// board.
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
    fn get_valid_moves(&self) -> Vec<RCPMove> {
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
    fn enter_move(&mut self, move_: RCPMove) -> Result<GameState, &str> {
        let (row, col, player) = move_;

        if row >= self.size || col >= self.size {
            return Err(OUT_OF_RANGE);
        }
        let idx = row * self.size + col;
        match &self.cells[idx] {
            Empty => self.cells[idx] = Full(player),
            OutOfBounds => return Err(OUT_OF_BOUNDS),
            _ => return Err(CELL_TAKEN),
        }
        self.move_history.push((row, col, player));

        if let Some(winner) = self.maybe_get_winner(move_) {
            return Ok(Ended(Winner(winner)));
        }

        if self.get_valid_moves().is_empty() {
            return Ok(Ended(Draw));
        }

        Ok(Ongoing)
    }

    fn move_history(&self) -> Vec<RCPMove> {
        self.move_history.clone()
    }
}
