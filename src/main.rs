/// Command line interface for playing tic-tac-toe against an AI.
use std::io;

use mcts_tic_tac_toe::EndState::{Draw, Winner};
use mcts_tic_tac_toe::GameState::{Ended, Ongoing};
use mcts_tic_tac_toe::Player::{P1, P2};
use mcts_tic_tac_toe::{HumanAgent, MonteCarloAgent, TicTacToeAgent, TicTacToeBoard, BOARD_SIZE};

fn main() -> io::Result<()> {
    let mut board = TicTacToeBoard::new(BOARD_SIZE);
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
