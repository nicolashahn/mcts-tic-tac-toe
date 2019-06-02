use std::io;

#[derive(Clone)]
enum Cell {
    Empty,
    User,
    Comp,
}

struct Board {
    cells: Vec<Cell>,
}

impl Board {
    fn new() -> Board {
        Board {
            cells: vec![Cell::Empty; 9],
        }
    }
    fn ended(&self) -> bool {
        // TODO use filter
        for c in &self.cells {
            if let Cell::Empty = c {
                return false;
            }
        }
        true
    }
}

fn get_move() -> io::Result<String> {
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let player_move = input.trim().to_string();

    Ok(player_move)
}

fn main() -> io::Result<()> {
    let board = Board::new();
    loop {
        let player_move = get_move()?;
        // TODO game logic
        println!("You typed: {}", player_move);

        if board.ended() {
            println!("Game over");
            return Ok(());
        }
    }
}
