#![allow(unused_imports)]

/// Command line interface for playing tic-tac-toe against an AI.
use std::io;

use mcts::{agents, board_game, fournotthree, play, tictactoe};

use agents::{ForgetfulSearchAgent, HumanAgent, MCTSAgent, RandomAgent};
use board_game::{GameBoard, Player};
use fournotthree::FourNotThreeBoard;
use tictactoe::TicTacToeBoard;

const BOARD_SIZE: usize = 3;
// number of random games to play out from a given game state each turn
// stop after we reach or exceed this number
const PLAYOUT_BUDGET: usize = 1_000_000;

// TODO add command line flags to control board size, player agent types, playout budget
fn main() -> io::Result<()> {
    //let board = TicTacToeBoard::new(BOARD_SIZE);
    let board = FourNotThreeBoard::new(BOARD_SIZE);

    let agent1 = HumanAgent::new(Player::P1);
    // Uncomment the following line to see two AI agents play each other:
    //let agent1 = MCTSAgent::new(Player::P1, PLAYOUT_BUDGET, board.clone());

    let agent2 = MCTSAgent::new(Player::P2, PLAYOUT_BUDGET, board.clone());

    play(agent1, agent2, board)
}
