use std::collections::HashMap;

use crate::base::node::generate_deriv_nodes;

use super::{Idx, Operation, Session};

pub struct Graph {
    nodes: Vec<Idx>,
    out_nodes: Vec<Idx>,
}

fn _dfs(idx: &Idx, nodes: &mut Vec<Idx>, session: &Session) {
    if !nodes.contains(idx) {
        let node = session.get_node(idx);
        // get children
        let children = node.get_input_nodes();
        for child in children {
            _dfs(&child, nodes, session)
        }

        nodes.push(*idx)
    }
}

impl Graph {
    pub fn construct(out_nodes: Vec<Idx>, session: &Session) -> Self {
        let mut nodes: Vec<Idx> = Vec::new();

        for out in out_nodes.iter() {
            // _dfs makes this automatically sorted
            _dfs(out, &mut nodes, session);
        }

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
}
