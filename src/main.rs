/// Monte Carlo tree search Tic-Tac-Toe agent and command line interface for playing against it.
use std::io;
use std::ops::Add;

const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

// error messages
const BAD_INPUT: &str = "bad input";
const OUT_OF_RANGE: &str = "out of range";
const CELL_TAKEN: &str = "cell taken";

/// At a given game state, how many wins are possible out of the total number of playouts?
/// Wins get score of 1, draws 0, losses -1
#[derive(Debug, PartialEq, PartialOrd)]
struct Outcomes {
    wins: isize,
    total: isize,
}

impl Outcomes {
    fn new(wins: isize, total: isize) -> Outcomes {
        Outcomes { wins, total }
    }
    fn as_f64(&self) -> f64 {
        return self.wins as f64 / self.total as f64;
    }
}

impl Add for Outcomes {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            wins: self.wins + other.wins,
            total: self.total + other.total,
        }
    }
}

/// The type of player
#[derive(Clone, Copy, Debug, PartialEq)]
enum Player {
    // You
    Human,
    // MonteCarloAgent (or other AI agent)
    AI,
}

impl Player {
    /// Get the opposite type (opponent) of this player
    fn get_opp(&self) -> Player {
        match self {
            Player::Human => Player::AI,
            Player::AI => Player::Human,
        }
    }
}

/// Represents a single cell of the tic-tac-toe board
#[derive(Clone, Copy, Debug)]
enum Cell {
    Empty,
    Full(Player),
}

/// Store the size and state of the tic-tac-toe board
#[derive(Clone, Debug)]
struct Board {
    // dimension of the board (total number of cells = size * size)
    size: usize,
    // for example: 3x3 grid would be a vec of length 9
    cells: Vec<Cell>,
}

// TODO eventually store interior node scores here so we don't need to check entire tree of
// possible games at every turn
struct MonteCarloAgent {}

impl MonteCarloAgent {
    fn new() -> MonteCarloAgent {
        MonteCarloAgent {}
    }

    fn get_valid_moves(&self, board: &Board) -> Vec<(usize, usize)> {
        let mut valid_moves = Vec::new();
        for (i, cell) in board.cells.iter().enumerate() {
            if let Cell::Empty = cell {
                valid_moves.push((i / board.size, i % board.size));
            }
        }
        return valid_moves;
    }

    /// Scores a given move by playing it out recursively alternating between the AI and Human
    /// players taking turns until it reaches all end states
    /// return (number of wins/number of overall end states)
    fn score_move(&self, board: &Board, player: Player, r: usize, c: usize) -> Outcomes {
        // create a new board with the move in question played
        let mut new_board = board.clone();
        if let Ok(Some(_)) = new_board.enter_move(r, c, player) {
            // if someone won, return the score
            return match player {
                Player::AI => Outcomes::new(1, 1),
                Player::Human => Outcomes::new(0, 1),
            };
        }

        // if this is an intermediate node:
        // get next possible moves for the opposing player
        let valid_moves = self.get_valid_moves(board);
        let opp = player.get_opp();

        // recurse to the possible subsequent moves and score them
        let mut total = Outcomes::new(0, 0);
        for (r, c) in &valid_moves {
            total = total + self.score_move(&new_board, opp, *r, *c);
        }

        // score for this move = # of wins / # of possible end states that stem from this move
        total
    }

    /// Agent chooses the best available move
    fn choose_move(&self, board: &Board) -> (usize, usize) {
        let valid_moves = self.get_valid_moves(board);

        let mut max_score = Outcomes::new(0, 1);
        let mut best_rc = valid_moves[0];
        for (r, c) in valid_moves {
            let score = self.score_move(board, Player::AI, r, c);
            println!("{} {} {:?}", r, c, score);
            if score.as_f64() > max_score.as_f64() {
                best_rc = (r, c);
                max_score = score;
            }
        }

        best_rc
    }
}

impl Board {
    /// Return a new Board of (size * size) cells
    fn new(size: usize) -> Board {
        Board {
            cells: vec![Cell::Empty; (size * size) as usize],
            size,
        }
    }

    /// Print the board to stdout with the column and row labels:
    ///
    ///   abc
    /// 0 ...
    /// 1 ...
    /// 2 ...
    ///
    /// NOTE: this will fail with self.size > 10, need to implement smarter indexing
    fn display(&self) {
        if self.size > 10 {
            println!("The board is too large to print with indices");
            return;
        }
        println!("\n  {}", &ALPHABET.to_string()[..self.size]);
        for i in 0..self.size {
            let mut row = String::with_capacity(self.size as usize);
            row.push_str(&format!("{:<2}", i));
            for j in 0..self.size {
                row.push(match &self.cells[i * self.size + j] {
                    Cell::Empty => '.',
                    Cell::Full(player) => match player {
                        Player::Human => 'o',
                        Player::AI => 'x',
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
/// Columns are letter indexes, rows are integers
/// Example: "a2" means column 0, row 2
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

fn main() -> io::Result<()> {
    let mut board = Board::new(3);

    println!("IT'S TIC-TAC-TOEEEEEEE TIIIIIIIIME!!!!!!");
    board.display();

    let ai = MonteCarloAgent::new();

    loop {
        println!("Enter a move (like \"a0\"):");

        let (row, col) = match get_move() {
            Ok((row, col)) => (row, col),
            Err(_) => {
                println!("Oops, enter valid input");
                continue;
            }
        };

        match board.enter_move(row, col, Player::Human) {
            Ok(Some(_)) => {
                board.display();
                println!("Game over, you are the winner!");
                return Ok(());
            }
            // show board and continue to AI's move
            Ok(None) => board.display(),
            // back to start of loop to let user enter a different move
            Err(msg) => {
                println!("Oops, illegal move: {}", msg);
                continue;
            }
        };

        println!("AI is thinking...");
        let (row, col) = ai.choose_move(&board);

        match board.enter_move(row, col, Player::AI) {
            Ok(Some(_)) => {
                board.display();
                println!("Game over, you lost!");
                return Ok(());
            }
            _ => board.display(),
        };
    }
}
