/// Agents for TicTacToe.
extern crate rand;

use std::collections::HashMap;
use std::f64;
use std::io;
use std::ops::Add;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::tic_tac_toe;

use tic_tac_toe::EndState::{Draw, Winner};
use tic_tac_toe::GameState::{Ended, Ongoing};
use tic_tac_toe::Player::{P1, P2};
use tic_tac_toe::{EndState, Player, TicTacToeAgent, TicTacToeBoard, ALPHABET};

const BAD_INPUT: &str = "bad input";

// From UCT formula, "theoretically equivalent to sqrt(2)" - see
// https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
const DEFAULT_EXPLORATION_CONSTANT: f64 = f64::consts::SQRT_2;

// These may need to be tweaked
const WIN_REWARD: isize = 1;
const DRAW_REWARD: isize = 0;
const LOSS_REWARD: isize = -1;

/// At a given game state, the summed wins/losses/draw scores, as well as the total number of
/// playouts that have been tried.
#[derive(Clone, Debug, PartialEq)]
struct Outcomes {
    score: isize,
    total: usize,
}

impl Outcomes {
    fn new(score: isize, total: usize) -> Outcomes {
        Outcomes { score, total }
    }
}

impl Add for Outcomes {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            score: self.score + other.score,
            total: self.total + other.total,
        }
    }
}

/// An agent controlled by the user running the program.
#[derive(Default)]
pub struct HumanAgent {}

impl TicTacToeAgent for HumanAgent {
    fn choose_move(&mut self, _board: &TicTacToeBoard) -> (usize, usize) {
        loop {
            println!("Enter a move (like \"a0\"):");
            match self.get_move() {
                Ok(rc) => return rc,
                Err(_) => {
                    println!("Oops, enter valid input");
                }
            };
        }
    }
}

impl HumanAgent {
    /// Accept player input from stdin, parse into (row, col) indexes.
    /// Columns are letter indexes, rows are integers.
    /// Example: "a2" means column 0, row 2
    fn get_move(&self) -> Result<(usize, usize), &'static str> {
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
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
}

#[derive(Clone, Debug)]
/// AI agent that plays using tree search to choose moves - but does not remember tree expansions
/// that have been calculated in prior moves. Not a true Monte Carlo tree search agent, BUT it's
/// blazing fast at raw playout computation.
pub struct ForgetfulSearchAgent {
    player: Player,
    playout_budget: usize,
}

impl TicTacToeAgent for ForgetfulSearchAgent {
    /// Agent chooses the best available move
    #[allow(clippy::explicit_counter_loop)]
    fn choose_move(&mut self, board: &TicTacToeBoard) -> (usize, usize) {
        println!("{:?} ForgetfulSearchAgent is thinking...", self.player);
        let valid_moves = board.get_valid_moves();
        let num_moves = valid_moves.len();

        let mut max_score = -((2 as isize).pow(62));
        let mut total_playouts = 0;
        let mut best_rowcol = valid_moves[0];

        let (sender, receiver) = mpsc::channel();

        let now = Instant::now();
        for (row, col) in valid_moves {
            // need a mutable copy here so we can use recursive backtracking without needing to make
            // a copy of the board at each step
            let mut theoretical_board = board.clone();
            let theoretical_self = self.clone();
            let theoretical_player = self.player.clone();
            let new_sender = sender.clone();
            // our "playout budget" for a single move is the total budget split evenly
            // between all the current possible moves
            let move_budget = self.playout_budget / num_moves;
            thread::spawn(move || {
                let outcomes = theoretical_self.score_move(
                    &mut theoretical_board,
                    theoretical_player,
                    row,
                    col,
                    move_budget,
                );
                new_sender.send((outcomes, row, col)).unwrap();
            });
        }

        let mut threads_finished = 0;
        for (outcomes, row, col) in receiver {
            threads_finished += 1;

            println!("Evaluating move (row {}, col {}), {:?}", row, col, outcomes);

            if outcomes.score > max_score {
                best_rowcol = (row, col);
                max_score = outcomes.score;
            }

            total_playouts += outcomes.total;

            if threads_finished == num_moves {
                break;
            }
        }

        println!(
            "
Chosen move:      {:?}
Total playouts:   {}
Choosing took:    {:?}
Playout rate:     {:.2}/sec",
            best_rowcol,
            total_playouts,
            now.elapsed(),
            (total_playouts as f64 / (now.elapsed().as_nanos() as f64)) * 1_000_000_000.0
        );

        best_rowcol
    }
}

impl ForgetfulSearchAgent {
    pub fn new(player: Player, playout_budget: usize) -> ForgetfulSearchAgent {
        ForgetfulSearchAgent {
            player,
            playout_budget,
        }
    }

    /// Scores a given move by playing it out on a theoretical board alternating between the agent and
    /// the opponent player taking turns (by recursively calling itself) until it reaches an end
    /// state as many times as it can before it reaches its playout_threshold.
    fn score_move(
        &self,
        board: &mut TicTacToeBoard,
        player: Player,
        row: usize,
        col: usize,
        playout_budget: usize,
    ) -> Outcomes {
        // play the move in question on the theoretical board
        if let Ok(Ended(endstate)) = board.enter_move(row, col, player) {
            // backtrack once we're done calculating
            if board.undo_move().is_err() {
                panic!("ForgetfulSearchAgent tried to do an illegal undo_move() at game end, board: {:?}", board);
            }

            // the score (num cells remaining)^2 + 1 weights outcomes that are closer in the search
            // space to the current move higher
            // end states where the board is fuller are less likely
            let cells_remaining = board.get_valid_moves().len();
            let score = cells_remaining * cells_remaining + 1;

            // return score/2 if win, -score if lose, 0 if draw
            // a win is only worth half as much as a loss because:
            //
            //      "To win, first you must not lose."
            //      - Nicolas Hahn
            //
            return match (endstate, self.player) {
                (Winner(P1), P1) | (Winner(P2), P2) => Outcomes::new((score / 2) as isize, 1),
                (Draw, _) => Outcomes::new(0, 1),
                _ => Outcomes::new(-(score as isize), 1),
            };
        }

        // if this is an intermediate node:
        // get next possible moves for the opposing player
        let mut valid_moves = board.get_valid_moves();
        let mut rng = thread_rng();
        valid_moves.shuffle(&mut rng);
        let opp = player.get_opponent();

        // recurse to the possible subsequent moves and score them
        let mut outcomes = Outcomes::new(0, 0);
        for (new_r, new_c) in &valid_moves {
            outcomes = outcomes + self.score_move(board, opp, *new_r, *new_c, playout_budget);

            if outcomes.total >= playout_budget {
                // we've met or surpassed the total # of games we're supposed to play out
                break;
            }
        }

        // backtrack once we're done calculating
        if !board.move_history.is_empty() && board.undo_move().is_err() {
            panic!("ForgetfulSearchAgent tried to do an illegal undo_move() while unwinding simulation");
        }

        outcomes
    }
}

#[derive(Clone, Debug)]
/// Agent that makes random moves.
pub struct RandomAgent {
    player: Player,
}

impl TicTacToeAgent for RandomAgent {
    fn choose_move(&mut self, board: &TicTacToeBoard) -> (usize, usize) {
        Self::get_random_move_choice(board)
    }
}

impl RandomAgent {
    pub fn new(player: Player) -> RandomAgent {
        RandomAgent { player }
    }

    pub fn get_random_move_choice(board: &TicTacToeBoard) -> (usize, usize) {
        let valid_moves = board.get_valid_moves();
        let mut rng = thread_rng();
        *valid_moves.choose(&mut rng).unwrap()
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Tree node for the Monte Carlo search tree.
pub struct TreeNode {
    // theoretical copy of the actual game board with the move the node represents applied
    board: TicTacToeBoard,
    // the player that made the most recent move, putting the board in its current state
    player: Player,
    // is this an end state for the game?
    is_end_state: bool,
    // have all child nodes been fully expanded?
    is_fully_expanded: bool,
    // number of games that have been played out from this node
    visits: usize,
    // total summed score for those games, based on number of wins/losses/draws
    score: isize,
    // mapping of the valid moves from this board state to the child TreeNodes
    // this should be empty if the node represents an end state
    children: HashMap<(usize, usize), TreeNode>,
}

impl TreeNode {
    /// Create a new TreeNode from the a parent node, and a move: copy the parent's board,
    /// apply the move, and then set all other fields appropriately. If the move did not result in
    /// an end state, simulate a random game and return the result as a score
    fn from_expansion(parent: &TreeNode, move_: (usize, usize)) -> (TreeNode, EndState) {
        let mut board = parent.board.clone();
        let player = parent.player.get_opponent();
        let (row, col) = move_;
        let (is_end_state, endstate) = match board.enter_move(row, col, player) {
            Ok(Ongoing) => (
                false,
                TreeNode::simulate_random_playout(&mut board.clone(), player.get_opponent()),
            ),
            Ok(Ended(endstate)) => (true, endstate),
            Err(msg) => panic!(
                "error in TreeNode.from_expansion when calling board.enter_move(): {}",
                msg
            ),
        };
        let mut node = TreeNode {
            board,
            player,
            is_end_state,
            is_fully_expanded: is_end_state,
            visits: 0,
            score: 0,
            children: HashMap::new(),
        };
        node.update_from_endstate(endstate);

        (node, endstate)
    }

    /// Play out a game randomly from this node and return the score.
    fn simulate_random_playout(theoretical_board: &mut TicTacToeBoard, player: Player) -> EndState {
        let mut curr_player = player.get_opponent();
        loop {
            let (row, col) = RandomAgent::get_random_move_choice(&theoretical_board);
            let game_state = theoretical_board.enter_move(row, col, curr_player);
            curr_player = curr_player.get_opponent();
            match game_state {
                Ok(state) => match state {
                    Ongoing => (),
                    Ended(endstate) => return endstate,
                },
                Err(msg) => panic!("Err in TreeNode.simulate_random_playout(): {}", msg),
            };
        }
    }

    /// This node or child of this node returned an EndState and passed it back up the tree.
    /// Update this node's visits and score appropriately.
    fn update_from_endstate(&mut self, endstate: EndState) {
        self.visits += 1;
        self.score += match endstate {
            Winner(winner) => {
                if winner == self.player {
                    WIN_REWARD
                } else {
                    LOSS_REWARD
                }
            }
            Draw => DRAW_REWARD,
        };
    }

    /// If all the node's children have been expanded, set the node as fully expanded.
    fn update_fully_expanded(&mut self) {
        if self.children.len() == self.board.get_valid_moves().len()
            && self
                .children
                .values()
                .filter(|&c| !c.is_fully_expanded)
                .collect::<Vec<&TreeNode>>()
                .is_empty()
        {
            self.is_fully_expanded = true;
        }
    }

    /// One round of tree expansion. Follow a path down until we get to a leaf that is not an
    /// end state, then playout until we hit an end state, creating more nodes as we go, and
    /// updating the parents back up after we reach the end state node.
    /// TODO use UCT formula to choose child to expand
    fn expand(&mut self) -> EndState {
        if self.is_fully_expanded {
            // TODO this is a hack, do something more correct here
            return Draw;
        }

        let mut rng = thread_rng();
        let mut moves = self.board.get_valid_moves();
        moves.shuffle(&mut rng);
        for move_ in moves.iter() {
            if self.children.get(&move_).is_none() {
                // we have a move that we don't have a child node for
                let (child_node, endstate) = TreeNode::from_expansion(&self, *move_);
                self.children.insert(*move_, child_node);
                self.update_from_endstate(endstate);
                self.update_fully_expanded();
                return endstate;
            }
        }

        // if we got here, all children have been visited, so we need to visit their children
        for move_ in moves.iter() {
            let child = self.children.get_mut(&move_).unwrap();
            if !child.is_fully_expanded {
                let endstate = child.expand();
                self.update_from_endstate(endstate);
                self.update_fully_expanded();
                return endstate;
            }
        }

        panic!("Should never be able to reach this");
    }

    #[allow(dead_code)]
    /// Debugging helper to see the size of the search tree.
    fn size_of_tree(&self) -> usize {
        if self.children.is_empty() {
            return 1;
        }
        self.children
            .values()
            .try_fold(0usize, |acc, c| acc.checked_add(c.size_of_tree()))
            .unwrap()
    }
}

#[derive(Clone, Debug)]
/// Monte Carlo tree search agent. Based on:
/// https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
pub struct MCTSAgent {
    player: Player,
    playout_budget: usize,
    root: TreeNode,
    exploration_constant: f64,
}

impl<'a> TicTacToeAgent for MCTSAgent {
    /// Agent chooses the best available move
    fn choose_move(&mut self, board: &TicTacToeBoard) -> (usize, usize) {
        println!("{:?} MCTSAgent is thinking...", self.player);
        self.search(&board)
    }
}

impl MCTSAgent {
    pub fn new(player: Player, playout_budget: usize, board_copy: TicTacToeBoard) -> Self {
        let root = TreeNode {
            board: board_copy,
            player: player.get_opponent(),
            is_end_state: false,
            is_fully_expanded: false,
            visits: 0,
            score: 0,
            children: HashMap::new(),
        };
        MCTSAgent {
            player,
            playout_budget,
            root,
            exploration_constant: DEFAULT_EXPLORATION_CONSTANT,
        }
    }

    /// Expand the tree for as many iterations as we can, then pick the best move thus far.
    fn search(&mut self, board: &TicTacToeBoard) -> (usize, usize) {
        let maybe_opp_move = self.get_opponents_last_move(&board);
        if let Some(opp_move) = maybe_opp_move {
            self.update_with_opponents_move(opp_move, &board);
        }
        // TODO multithread
        for _ in 0..self.playout_budget {
            if self.root.is_fully_expanded {
                break;
            }
            let _ = self.root.expand();
        }

        self.get_best_move_and_promote_child()
    }

    /// Get the last move of the opponent, if we're not making the first move on the board.
    fn get_opponents_last_move(&self, board: &TicTacToeBoard) -> Option<(usize, usize)> {
        for i in 0..board.cells.len() {
            if self.root.board.cells.get(i) != board.cells.get(i) {
                return Some((i / board.size, i % board.size));
            }
        }

        None
    }

    /// Update the root node to reflect the current state of the game by promoting the child node
    /// that represents the move the opponent just made to root, or just updating the board if we
    /// have no children.
    fn update_with_opponents_move(&mut self, opp_move: (usize, usize), board: &TicTacToeBoard) {
        match self.root.children.remove(&opp_move) {
            Some(child_node) => self.root = child_node,
            None => self.root.board = board.clone(),
        }
    }

    /// Get the move for the child with the highest score for this node (representing the best move
    /// we can make from this state). Uses the expression found here to calculate node value:
    /// https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
    /// Also promote the child with the highest score to the root of the tree.
    #[allow(clippy::float_cmp)]
    fn get_best_move_and_promote_child(&mut self) -> (usize, usize) {
        let mut best_val = -(f64::powf(2., 63.));
        let mut best_moves: Vec<(usize, usize)> = vec![];
        for (&move_, child) in self.root.children.iter() {
            let node_val = child.score as f64 / child.visits as f64;
            println!("{:?} score: {}", child.board, node_val);
            if node_val > best_val {
                best_val = node_val;
                best_moves = vec![move_];
            } else if node_val == best_val {
                best_moves.push(move_);
            }
        }
        let mut rng = thread_rng();
        let &best_move = best_moves.choose(&mut rng).unwrap();
        let new_root = self.root.children.remove(&best_move).unwrap();
        self.root = new_root;

        best_move
    }
}
