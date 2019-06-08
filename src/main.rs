/// Monte Carlo tree search Tic-Tac-Toe agent and command line interface for playing against it.
use std::io;

use mcts_tic_tac_toe::EndState::{Draw, Winner};
use mcts_tic_tac_toe::GameState::{Ended, Ongoing};
use mcts_tic_tac_toe::Player::{P1, P2};
use mcts_tic_tac_toe::{Board, HumanAgent, MonteCarloAgent, TicTacToeAgent};

// TODO add command line flags to control board size, player agent types, playout budget
pub const BOARD_SIZE: usize = 4;

fn main() -> io::Result<()> {
    let mut board = Board::new(BOARD_SIZE);
    //let agent1 = MonteCarloAgent::new(P1);
    let agent1 = HumanAgent::new();
    let agent2 = MonteCarloAgent::new(P2);

    println!("\nIT'S TIC-TAC-TOEEEEEEE TIIIIIIIIME!!!!!!");
    board.display();

    loop {
        let player = match board.is_p1_turn {
            true => P1,
            false => P2,
        };
        let rc = match player {
            P1 => agent1.choose_move(&board),
            P2 => agent2.choose_move(&board),
        };

        let (row, col) = rc;
        match board.enter_move(row, col, player) {
            Ok(Ended(endstate)) => {
                board.display();
                match endstate {
                    Winner(player) => println!("Game over, {:?} won!", player),
                    Draw => println!("Game over, it's a draw!"),
                }
                return Ok(());
            }
            Ok(Ongoing) => board.display(),
            Err(msg) => {
                println!("Oops, illegal move: {}", msg);
            }
        };
    }
}
