use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use crate::node::generate_deriv_nodes;

use super::{Idx, Operation, Session};

// static mut total_calls: usize = 0;
// static mut total_not_contained: usize = 0;

pub struct Graph {
    nodes: Vec<Idx>,
    out_nodes: Vec<Idx>,
}

// NOTE: This call takes long and isn't necessary as Index is automaticlly sorted
//
fn _dfs(idx: &Idx, nodes: &mut HashSet<Idx>, session: &Session) {
    // Options to Optimize this step
    // - Make the Values Tensors
    // - Enable Shared Inputs (Cuts out on duplicate checks)
    // - Leave as is, as end goal doesn't use Tensors
    // unsafe {
    //     total_calls += 1;
    //     println!(
    //         "dfs: {}/{}, nodes_len: {}",
    //         total_not_contained,
    //         total_calls,
    //         nodes.len()
    //     );
    // }
    //
    let children = session.get_node(idx).get_input_nodes();
    if !nodes.contains(idx) {
        // unsafe {
        //     total_not_contained += 1;
        // }
        if let Some(children) = children {
            for child in children {
                _dfs(&child, nodes, session);
            }
        }
        nodes.insert(*idx);
    }
}

impl Graph {
    pub fn construct(out_nodes: Vec<Idx>, session: &Session) -> Self {
        let mut nodes: HashSet<Idx> = HashSet::new();

        for out in out_nodes.iter() {
            // NOTE: Not necessary
            // _dfs makes this automatically sorted
            _dfs(out, &mut nodes, session);
        }

        let mut nodes: Vec<Idx> = nodes.iter().cloned().collect();
        nodes.sort();

        Graph { nodes, out_nodes }
    }

    pub fn evaluate(&self, current_values: &mut HashMap<Idx, f32>, session: &Session) -> Vec<f32> {
        for node in self.nodes.iter() {
            if current_values.contains_key(node) {
                continue;
            }
            let value = session.get_node(node).evaluate(current_values);
            current_values.insert(*node, value);
        }

        self.out_nodes
            .iter()
            .map(|idx| current_values[idx])
            .collect()
    }

    pub fn generate_deriv_graph(&self, session: &mut Session) -> (Graph, HashMap<Idx, Idx>) {
        let mut deriv_nodes: HashMap<Idx, Idx> = HashMap::with_capacity(self.nodes.len());
        self.out_nodes.iter().for_each(|idx| {
            deriv_nodes.insert(*idx, session.add_node(Operation::Constant(1.)));
        });
        for idx in self.nodes.iter().rev() {
            // Generate all input deriv for idx
            let operation = session.get_node(idx).get_operation();
            let partial_derivs = generate_deriv_nodes(operation, deriv_nodes[idx], session);
            if let Some(partial_derivs) = partial_derivs {
                for (idx, deriv_idx) in partial_derivs.iter() {
                    deriv_nodes.insert(
                        *idx,
                        if deriv_nodes.contains_key(idx) {
                            session.add_node(Operation::Sum(*deriv_idx, deriv_nodes[idx]))
                        } else {
                            *deriv_idx
                        },
                    );
                }
            }
        }

        (
            Graph::construct(deriv_nodes.values().cloned().collect(), session),
            deriv_nodes,
        )
    }

    pub fn get_n_nodes(&self) -> usize {
        self.nodes.len()
    }
}
