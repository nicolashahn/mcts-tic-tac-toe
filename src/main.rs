/// Command line interface for playing tic-tac-toe against an AI.
use std::io;

#[allow(unused_imports)]
use agents::{ForgetfulSearchAgent, HumanAgent, MCTSAgent};
use mcts::{agents, play, tic_tac_toe};
use tic_tac_toe::{Player, TicTacToeBoard};

const BOARD_SIZE: usize = 5;
// number of random games to play out from a given game state
// stop after we reach or exceed this number
const PLAYOUT_BUDGET: usize = 1000000;

// TODO add command line flags to control board size, player agent types, playout budget
fn main() -> io::Result<()> {
    let mut board = TicTacToeBoard::new(BOARD_SIZE);
    let mut agent1 = HumanAgent::default();
    // To see two AI agents duel each other:
    //let agent1 = ForgetfulSearchAgent::new(Player::P1, PLAYOUT_BUDGET);
    let mut agent2 = ForgetfulSearchAgent::new(Player::P2, PLAYOUT_BUDGET);
    //let mut agent2 = MCTSAgent::new(Player::P2, PLAYOUT_BUDGET, board.clone());

    play(&mut agent1, &mut agent2, &mut board)
}
