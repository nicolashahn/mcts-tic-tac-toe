/// Generic board game interface.
use std::fmt;
use std::hash::Hash;

use Player::{P1, P2};

pub const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

// Error messages
pub const OUT_OF_RANGE: &str = "out of range";
pub const CELL_TAKEN: &str = "cell taken";
pub const NO_MOVE_TO_UNDO: &str = "no move to undo";

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

/// Used for deciding whose turn it is. P1 goes first.
#[derive(Eq, Hash, Clone, Copy, Debug, PartialEq)]
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

/// Represents a single cell of a game board.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cell {
    Empty,
    Full(Player),
}

/// A move in a (board) game, an action that an agent can take when it is their turn. For example,
/// a Tic Tac Toe move can be represented as (row, column, player).
pub trait GameMove: fmt::Debug + Send + Sync + Clone + Copy + Eq + Hash + 'static {
    /// Get the player that is making this move.
    fn player(&self) -> Player;
    fn set_player(&mut self, p: Player);
}

/// Functionality associated with any two player board-game-like object.
pub trait GameBoard<GM>: Clone + fmt::Debug + Send + Sync + std::marker::Sized + 'static
where
    GM: GameMove,
{
    /// Enter a move onto the board.
    fn enter_move(&mut self, move_: GM) -> Result<GameState, &str>;

    /// Get all the valid moves that are allowed at the board's current state.
    fn get_valid_moves(&self) -> Vec<GM>;

    /// Print a representation of the board to STDOUT.
    fn display(&self);

    /// Undo the last move that was made (for backtracking in tree search).
    fn undo_move(&mut self) -> Result<(), &str>;

    /// Get the history of the moves made in order (last move is most recent).
    fn move_history(&self) -> Vec<GM>;

    /// True if it's P1's turn, false if P2's.
    /// NOTE: This method you get for free by implementing `move_history()`, but a specific
    /// implementation for each concrete type can be much more performant.
    fn is_p1_turn(&self) -> bool {
        match self.move_history().last() {
            Some(move_) => move_.player() == P2,
            None => true,
        }
    }
}

/// Basic game move for any game that just allows player to lay down pieces on a grid.
/// (row, col, player)
pub type RCPMove = (usize, usize, Player);

impl GameMove for RCPMove {
    fn player(&self) -> Player {
        self.2
    }
    fn set_player(&mut self, p: Player) {
        self.2 = p;
    }
}
