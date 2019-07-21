use std::io;
pub mod agents;
pub mod board_games;

use board_games::EndState::{Draw, Winner};
use board_games::GameState::{Ended, Ongoing};
use board_games::Player::{P1, P2};
use board_games::{BoardGameAgent, GameBoard, GameMove};

/// Two agents (human or AI) play against each other.
pub fn play<GM: GameMove, GB: GameBoard<GM>>(
    mut agent1: impl BoardGameAgent<GM, GB>,
    mut agent2: impl BoardGameAgent<GM, GB>,
    mut board: GB,
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
