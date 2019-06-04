use std::io;

const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

// error messages
const BAD_INPUT: &str = "bad input";
const OUT_OF_RANGE: &str = "out of range";
const CELL_TAKEN: &str = "cell taken";

enum State {
    GameOver,
    Continue,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Player {
    User,
    Comp,
}

#[derive(Clone, Copy, Debug)]
enum Cell {
    Empty,
    Full(Player),
}

struct Board {
    cells: Vec<Cell>,
    size: usize,
}

impl Board {
    fn new(size: usize) -> Board {
        Board {
            cells: vec![Cell::Empty; (size * size) as usize],
            size,
        }
    }

    fn display(&self) {
        println!("\n  {}", &ALPHABET.to_string()[..self.size]);
        for i in 0..self.size {
            let mut row = String::with_capacity(self.size as usize);
            row.push_str(&format!("{:<2}", i));
            for j in 0..self.size {
                row.push(match &self.cells[i * self.size + j] {
                    Cell::Empty => '.',
                    Cell::Full(player) => match player {
                        Player::User => 'o',
                        Player::Comp => 'x',
                    },
                });
            }
            println!("{}", row);
        }
        println!("\n");
    }

    /// Return Ok(Some(player)) if game is over, Ok(None) if it continues, Err if invalid move
    fn enter_move(
        &mut self,
        row: usize,
        col: usize,
        player: Player,
    ) -> Result<Option<Player>, &str> {
        let idx = row * self.size + col;
        if idx > self.size * self.size - 1 {
            return Err(OUT_OF_RANGE);
        }
        match &self.cells[idx] {
            Cell::Empty => self.cells[idx] = Cell::Full(player),
            _ => return Err(CELL_TAKEN),
        }

        if self.move_wins_game(row, col, player) {
            return Ok(Some(player));
        }

        Ok(None)
    }

    /// Return if the line defined by the filter_fn is filled with cells of type player
    fn line_is_filled_with_player(&self, player: Player, filter_fn: &Fn(usize) -> bool) -> bool {
        let matching_cells: Vec<(usize, &Cell)> = self
            .cells
            .iter()
            .enumerate()
            .filter(|&(i, x)| {
                filter_fn(i)
                    && match x {
                        Cell::Full(cplayer) => &player == cplayer,
                        _ => false,
                    }
            })
            .collect();

        matching_cells.len() == self.size
    }

    /// Did the last move played at (r,c) by player win the game?
    fn move_wins_game(&self, r: usize, c: usize, player: Player) -> bool {
        // check row
        if self.line_is_filled_with_player(player, &|i| i / self.size == r) {
            return true;
        }

        // check col
        if self.line_is_filled_with_player(player, &|i| i % self.size == c) {
            return true;
        }

        // check \ diagonal if relevant
        if self.line_is_filled_with_player(player, &|i| i % self.size == i / self.size) {
            return true;
        }

        // check / diagonal if relevant
        if self.line_is_filled_with_player(player, &|i| {
            (i % self.size) + (i / self.size) == self.size - 1
        }) {
            return true;
        }

        false
    }
}

/// Accept player input from stdin, parse into (row, col) indexes
fn get_move() -> Result<(usize, usize), &'static str> {
    let mut input = String::new();
    if let Err(_) = io::stdin().read_line(&mut input) {
        return Err(BAD_INPUT);
    }
    // at least 2 indices + \n
    if input.len() < 3 {
        return Err(BAD_INPUT);
    }
    let player_move = input.trim();

    let col = match ALPHABET.find(player_move.chars().nth(0).unwrap()) {
        Some(idx) => idx,
        None => return Err(BAD_INPUT),
    };

    let row = match player_move.chars().nth(1).unwrap().to_digit(10) {
        Some(idx) => idx as usize,
        None => return Err(BAD_INPUT),
    };

    Ok((row, col))
}

fn handle_move_result(move_result: Result<Option<Player>, &str>) -> Result<State, &str> {
    match move_result {
        Ok(Some(player)) => {
            println!("Game over, {:?} is the winner!", player);
            return Ok(State::GameOver);
        }
        Ok(None) => (),
        Err(msg) => {
            println!("Oops, illegal move: {}", msg);
            return Err(msg);
        }
    }
    Ok(State::Continue)
}

fn main() -> io::Result<()> {
    let mut board = Board::new(3);

    println!("IT'S TIC-TAC-TOEEEEEEE TIIIIIIIIMEEEE!!!!!!");
    loop {
        board.display();
        println!("Enter a move (like \"a0\"):");

        let (row, col) = match get_move() {
            Ok((row, col)) => (row, col),
            Err(_) => {
                println!("Oops, enter valid input");
                continue;
            }
        };

        let move_result = board.enter_move(row, col, Player::User);
        match handle_move_result(move_result) {
            Ok(State::GameOver) => return Ok(()),
            Ok(State::Continue) => (),
            // go back to beginning of loop and let player enter another move
            Err(_) => continue,
        }

        // TODO computer moves
    }
}
