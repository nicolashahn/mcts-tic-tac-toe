use std::io;
pub mod agents;
pub mod board_game;
pub mod fournotthree;
pub mod tictactoe;

use agents::BoardGameAgent;
use board_game::EndState::{Draw, Winner};
use board_game::GameState::{Ended, Ongoing};
use board_game::Player::{P1, P2};
use board_game::{GameBoard, GameMove};

/// Two agents (human or AI) play against each other.
#[allow(clippy::match_bool)]
pub fn play<GM: GameMove, GB: GameBoard<GM>>(
    mut agent1: impl BoardGameAgent<GM, GB>,
    mut agent2: impl BoardGameAgent<GM, GB>,
    mut board: GB,
) -> io::Result<()> {
    println!("\nStarting a game");
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
