/// Command line interface for playing tic-tac-toe against an AI.
use std::io;

use mcts::{agents, board_game, fournotthree, play, tictactoe};

#[allow(unused_imports)]
use agents::{ForgetfulSearchAgent, HumanAgent, MCTSAgent, RandomAgent};
use board_game::{GameBoard, Player};
use fournotthree::FourNotThreeBoard;
use tictactoe::TicTacToeBoard;

const BOARD_SIZE: usize = 3;
// number of random games to play out from a given game state
// stop after we reach or exceed this number
const PLAYOUT_BUDGET: usize = 500_000;

// TODO add command line flags to control board size, player agent types, playout budget
fn main() -> io::Result<()> {
    let board = FourNotThreeBoard::new(BOARD_SIZE);
    //let board = TicTacToeBoard::new(BOARD_SIZE);

    let agent1 = HumanAgent::new(Player::P1);
    // To see two AI agents duel each other:
    //let mut agent1 = ForgetfulSearchAgent::new(Player::P1, 2_000_000);
    //let mut agent1 = MCTSAgent::new(Player::P1, PLAYOUT_BUDGET, board.clone());

    //let agent2 = RandomAgent::new(Player::P2);
    //let agent2 = ForgetfulSearchAgent::new(Player::P2, PLAYOUT_BUDGET);
    let agent2 = MCTSAgent::new(Player::P2, PLAYOUT_BUDGET, board.clone());

    play(agent1, agent2, board)
}
