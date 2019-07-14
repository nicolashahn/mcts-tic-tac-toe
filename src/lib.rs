use std::io;
pub mod agents;
pub mod tic_tac_toe;

use tic_tac_toe::EndState::{Draw, Winner};
use tic_tac_toe::GameState::{Ended, Ongoing};
use tic_tac_toe::Player::{P1, P2};
use tic_tac_toe::{GameBoard, TicTacToeAgent, TicTacToeBoard};

/// Two agents (human or AI) play against each other.
pub fn play(
    agent1: &mut TicTacToeAgent,
    agent2: &mut TicTacToeAgent,
    board: &mut TicTacToeBoard,
) -> io::Result<()> {
    println!("\nIT'S TIC-TAC-TOEEEEEEE TIIIIIIIIME!!!!!!");
    board.display();

    loop {
        let player = match board.is_p1_turn() {
            true => P1,
            false => P2,
        };
        let move_ = match player {
            P1 => agent1.choose_move(&board),
            P2 => agent2.choose_move(&board),
        };

        match board.enter_move(move_) {
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
