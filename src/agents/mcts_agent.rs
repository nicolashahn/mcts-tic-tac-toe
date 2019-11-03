/// Monte Carlo tree search agent.
extern crate num_cpus;
extern crate scoped_threadpool;

use std::collections::HashMap;
use std::f64;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use scoped_threadpool::Pool;

use crate::agents::BoardGameAgent;
use crate::board_game;

use board_game::EndState::{Draw, Winner};
use board_game::GameState::{Ended, Ongoing};
use board_game::{EndState, GameBoard, GameMove, Player};

// From UCT formula, "theoretically equivalent to sqrt(2)" - see
// https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
const DEFAULT_EXPLORATION_CONSTANT: f64 = f64::consts::SQRT_2;

#[derive(Clone, Debug, PartialEq)]
/// Tree node for the Monte Carlo search tree.
pub struct TreeNode<GM, GB>
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    // Theoretical copy of the actual game board with the move the node represents applied.
    board: GB,
    // The player that made the most recent move, putting the board in its current state.
    player: Player,
    // Is this an end state for the game?
    is_end_state: bool,
    // Have all child nodes been fully expanded?
    is_fully_expanded: bool,
    // Number of games that have been played out from this node.
    visits: usize,
    // Number of each outcome possible from this node. Draws = visits - wins - losses
    wins: usize,
    losses: usize,
    // Mapping of the valid moves from this board state to the child TreeNodes. This should be
    // empty if the node represents an end state.
    children: HashMap<GM, TreeNode<GM, GB>>,
    // Exploration constant for UCT, inherited from MCTSAgent
    exploration_constant: f64,
}

impl<GM, GB> TreeNode<GM, GB>
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    /// Create a new TreeNode from the a parent node, and a move: copy the parent's board,
    /// apply the move, and then set all other fields appropriately. If the move did not result in
    /// an end state, simulate a random game and return the result as a score
    fn from_expansion(parent: &TreeNode<GM, GB>, move_: GM) -> (TreeNode<GM, GB>, EndState) {
        let mut board = parent.board.clone();
        let player = parent.player.get_opponent();
        let (is_end_state, endstate) = match board.enter_move(move_) {
            Ok(Ongoing) => (
                false,
                TreeNode::<GM, GB>::simulate_random_playout(
                    &mut board.clone(),
                    player.get_opponent(),
                ),
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
            wins: 0,
            losses: 0,
            children: HashMap::new(),
            exploration_constant: parent.exploration_constant,
        };
        node.update_from_endstate(endstate);

        (node, endstate)
    }

    /// Play out a game randomly from this node and return the score.
    fn simulate_random_playout(
        theoretical_board: &mut impl GameBoard<GM>,
        player: Player,
    ) -> EndState {
        let mut curr_player = player.get_opponent();
        loop {
            let mut rng = thread_rng();
            let &move_ = theoretical_board
                .get_valid_moves()
                .choose(&mut rng)
                .unwrap();
            let game_state = theoretical_board.enter_move(move_);
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
        if let Winner(winner) = endstate {
            if winner == self.player {
                self.wins += 1
            } else {
                self.losses += 1
            }
        }
    }

    /// If all the node's children have been expanded, set the node as fully expanded.
    fn update_fully_expanded(&mut self) {
        if self.children.len() == self.board.get_valid_moves().len()
            && self
                .children
                .values()
                .filter(|&c| !c.is_fully_expanded)
                .count()
                == 0
        {
            self.is_fully_expanded = true;
        }
    }

    /// Get the exploration score as defined by the UCT formula:
    /// http://mcts.ai/about/
    /// This gives us a weighting for how much effort we should expend exploring this node vs.
    /// others.
    fn get_exploration_score(&self, exploration_constant: f64, move_: &GM) -> f64 {
        // Ratio of simulations from the given node that we didn't lose
        let get_node_not_loss_ratio = |n: &TreeNode<GM, GB>| {
            1. - (n.losses as f64 / n.visits as f64) - (0.5 * (n.draws() as f64 / n.visits as f64))
        };
        // UCB/UCT formula
        let calculate_uct = |score: f64, c_visits: usize, p_visits: usize| {
            score + exploration_constant * (f64::ln(p_visits as f64) / c_visits as f64)
        };
        match self.children.get(&move_) {
            Some(child) => {
                if child.is_fully_expanded {
                    0.
                } else {
                    calculate_uct(get_node_not_loss_ratio(&child), child.visits, self.visits)
                }
            }
            None => exploration_constant,
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

        let mut moves = self.board.get_valid_moves();
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

        // use UCT to choose which nodes to expand
        moves.sort_by(|m1, m2| {
            self.get_exploration_score(self.exploration_constant, m1)
                .partial_cmp(&self.get_exploration_score(self.exploration_constant, m2))
                .unwrap()
        });

        while !moves.is_empty() {
            let best_move = moves.pop().unwrap();
            let child = self.children.get_mut(&best_move).unwrap();
            if !child.is_fully_expanded {
                let endstate = child.expand();
                self.update_from_endstate(endstate);
                self.update_fully_expanded();
                return endstate;
            }
        }

        panic!("TreeNode::expand() is broken");
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

    fn draws(&self) -> usize {
        self.visits - self.wins - self.losses
    }
}

#[derive(Clone, Debug)]
/// Monte Carlo tree search agent. Based on:
/// [pbsinclair42/MCTS](https://github.com/pbsinclair42/MCTS/blob/master/mcts.py)
pub struct MCTSAgent<GM, GB>
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    // Are we P1 or P2?
    player: Player,
    // Number of playout iterations we're allowed to simulate for each move choice.
    playout_budget: usize,
    // Root of the search tree. Will be replaced by one of its descendents upon choosing a move.
    root: TreeNode<GM, GB>,
    // Controls to what extent we should search unexplored vs explored (and well-scored) nodes.
    exploration_constant: f64,
}

impl<GM, GB> BoardGameAgent<GM, GB> for MCTSAgent<GM, GB>
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    fn choose_move(&mut self, board: &GB) -> GM {
        println!("{:?} MCTSAgent is thinking...\n", self.player);
        let theoretical_board = board.clone();
        self.search(&theoretical_board)
    }
}

impl<GM, GB> MCTSAgent<GM, GB>
where
    GM: GameMove,
    GB: GameBoard<GM>,
{
    pub fn new(player: Player, playout_budget: usize, board_copy: GB) -> Self {
        let root = TreeNode {
            board: board_copy,
            player: player.get_opponent(),
            is_end_state: false,
            is_fully_expanded: false,
            visits: 0,
            wins: 0,
            losses: 0,
            children: HashMap::new(),
            // TODO allow this to be passed in as a parameter
            exploration_constant: DEFAULT_EXPLORATION_CONSTANT,
        };
        MCTSAgent {
            player,
            playout_budget,
            root,
            exploration_constant: DEFAULT_EXPLORATION_CONSTANT,
        }
    }

    /// Update our state with the opponent's last move, expand the search tree, then promote the
    /// best child and return the best move.
    fn search(&mut self, board: &GB) -> GM {
        let now = Instant::now();
        let maybe_opp_move = self.get_opponents_last_move(board);
        if let Some(opp_move) = maybe_opp_move {
            self.update_with_opponents_move(opp_move, board);
        }

        // This is only guaranteed to create a new child while the expand() function chooses
        // unexplored moves first - change this if we choose another method for expansion
        let num_moves = board.get_valid_moves().len();
        while self.root.children.len() < num_moves {
            self.root.expand();
        }

        let mut pool = Pool::new(num_cpus::get() as u32);
        pool.scoped(|scoped| {
            let playout_budget = self.playout_budget;
            for child in self.root.children.values_mut() {
                scoped.execute(move || {
                    for _ in 0..playout_budget / num_moves {
                        child.expand();
                    }
                });
            }
        });

        let total_playouts: usize = self.root.children.values().map(|child| child.visits).sum();
        let best_move = self.get_best_move_and_promote_child();
        println!(
            "
Chosen move:      {:?}
Total playouts:   {}
Choosing took:    {:?}
Playout rate:     {:.2}/sec",
            best_move,
            total_playouts,
            now.elapsed(),
            (total_playouts as f64 / (now.elapsed().as_nanos() as f64)) * 1_000_000_000.0
        );

        best_move
    }

    /// Get the last move of the opponent, if we're not making the first move on the board.
    fn get_opponents_last_move(&self, board: &GB) -> Option<GM> {
        match board.move_history().last() {
            None => None,
            Some(&gm) => Some(gm),
        }
    }

    /// Update the root node to reflect the current state of the game by promoting the child node
    /// that represents the move the opponent just made to root, or just updating the board if we
    /// have no children.
    fn update_with_opponents_move(&mut self, opp_move: GM, board: &GB) {
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
    fn get_best_move_and_promote_child(&mut self) -> GM {
        let (mut best_win_r, mut best_loss_r) = (0., f64::powf(2., 64.));
        let mut best_moves: Vec<GM> = vec![];
        println!(
            "(move: visits (V), wins (W), losses (L), draws (D), win ratio (WR), loss ratio (LR), draw ratio (DR))"
        );
        for (&move_, child) in self.root.children.iter() {
            let (node_win_r, node_loss_r, node_draw_r) = (
                child.wins as f64 / child.visits as f64,
                child.losses as f64 / child.visits as f64,
                child.draws() as f64 / child.visits as f64,
            );
            println!(
                "{:?}: V: {}, W: {}, L: {}, D: {}, WR: {}, LR: {}, DR: {}",
                move_,
                child.visits,
                child.wins,
                child.losses,
                child.draws(),
                node_win_r,
                node_loss_r,
                node_draw_r,
            );
            // First, choose the move that leads to the fewest losses
            if node_loss_r < best_loss_r {
                best_loss_r = node_loss_r;
                best_win_r = node_win_r;
                best_moves = vec![move_];
            } else if node_loss_r == best_loss_r {
                // Then, if we have a tie, choose the one that leads to the most wins
                if node_win_r > best_win_r {
                    best_loss_r = node_loss_r;
                    best_win_r = node_win_r;
                    best_moves = vec![move_];
                } else if node_win_r == best_win_r {
                    best_moves.push(move_);
                }
            }
        }
        let mut rng = thread_rng();
        let &best_move = best_moves.choose(&mut rng).unwrap();
        let new_root = self.root.children.remove(&best_move).unwrap();
        self.root = new_root;

        best_move
    }
}

/// This test is under the assumption that (1,1) is indeed the best move to make, and that the
/// agent can always calculate it within 5000 playouts.
#[test]
fn test_mcts_agent_tic_tac_toe_first_move() {
    use crate::tictactoe;
    use board_game::Player::P1;
    use tictactoe::TicTacToeBoard;

    let board = TicTacToeBoard::new(3);
    let mut agent = MCTSAgent::new(P1, 5000, board.clone());
    let move_ = agent.search(&board);
    assert!(move_ == (1, 1, P1));
}
