/// 4not3 ("four!three"?) game interface.
///
/// Rules: The game is played on a hex grid usually of radius 6. Each player alternates laying down
/// a piece anywhere on the board (like TicTacToe). A player wins if they get four or more of their
/// pieces in a row, but loses if exactly three of their pieces are in a row.
use std::fmt;

use crate::board_game;

use board_game::Cell::{Empty, Full};
use board_game::EndState::{Draw, Winner};
use board_game::GameState::{Ended, Ongoing};
use board_game::Player::{P1, P2};
use board_game::{
    Cell, GameBoard, GameMove, GameState, Player, RCPMove, ALPHABET, CELL_TAKEN, NO_MOVE_TO_UNDO,
    OUT_OF_RANGE,
};

/// Store the size and state of the tic-tac-toe board.
#[derive(Clone, PartialEq)]
pub struct FourNotThreeBoard {
    // Dimension of the board (total number of cells = size * size)
    pub size: usize,
    // All cells in the hex grid, arranged like so:
    // 1 2 3
    //  4 5 6
    //   7 8 9
    // To achieve the hexagonal board shape, we need to set some cells as filled with neither
    // player (indexes 1 and 9 in this example)
    pub cells: Vec<Cell>,
    // Using the row, col in RCPMove as axial coordinates, see
    // https://www.redblobgames.com/grids/hexagons/#coordinates-cube
    pub move_history: Vec<RCPMove>,
}

impl FourNotThreeBoard {
    pub fn new(size: usize) -> FourNotThreeBoard {
        FourNotThreeBoard {
            cells: vec![Empty; (size * size) as usize],
            size,
            move_history: vec![],
        }
    }

    /// Get the cells that are adjacent to the one given. With a hex grid, there should be 6 cells
    /// returned, unless the given cell is at the edge of the grid.
    pub fn get_adjacents(rowcol: (usize, usize)) -> Vec<RCPMove> {
        // TODO
    }
}

impl GameBoard<XYZPMove> for FourNotThreeBoard {
    fn display(&self) {
        // TODO
    }

    fn is_p1_turn(&self) -> bool {
        // TODO
    }

    fn undo_move(&mut self) -> Result<(), &str> {
        // TODO
    }

    fn get_valid_moves(&self) -> Vec<XYZPMove> {
        // TODO
    }

    fn enter_move(&mut self, move_: XYZPMove) -> Result<GameState, &str> {
        // TODO
    }

    fn move_history(&self) -> Vec<XYZPMove> {
        self.move_history.clone()
    }
}
