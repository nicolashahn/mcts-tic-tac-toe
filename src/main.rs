/// Command line interface for playing tic-tac-toe against an AI.
use std::io;

use agents::{ForgetfulSearchAgent, HumanAgent};
use mcts::{agents, play, tic_tac_toe};
use tic_tac_toe::{Player, TicTacToeBoard};

const BOARD_SIZE: usize = 5;
// number of random games to play out from a given game state
// stop after we reach or exceed this number
const PLAYOUT_BUDGET: usize = 1_000_000;

// TODO add command line flags to control board size, player agent types, playout budget
fn main() -> io::Result<()> {
    let mut board = TicTacToeBoard::new(BOARD_SIZE);
    let agent1 = HumanAgent::new();
    // To see two AI agents duel each other:
    //let agent1 = ForgetfulSearchAgent::new(Player::P1, PLAYOUT_BUDGET);
    let agent2 = ForgetfulSearchAgent::new(Player::P2, PLAYOUT_BUDGET);

    play(&agent1, &agent2, &mut board)
}
