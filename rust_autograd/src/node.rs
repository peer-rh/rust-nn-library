use std::collections::HashMap;

use super::{Idx, Operation, Session};

pub struct Node {
    idx: Idx,
    operation: Operation,
    // requires_grad: bool
}

impl Node {
    pub fn new(own_idx: Idx, operation: Operation) -> Self {
        Node {
            idx: own_idx,
            operation,
        }
    }

    pub fn evaluate(&self, current_values: &mut HashMap<Idx, f32>) -> f32 {
        let out = self.operation.forward(current_values);
        self.conditional_debug(out);
        out
    }

    pub fn get_input_nodes(&self) -> Option<Vec<Idx>> {
        self.operation.get_input_nodes()
    }

    pub fn get_operation(&self) -> Operation {
        self.operation
    }

    fn conditional_debug(&self, out: f32) {
        // TODO: Find way to easily define way to define if it should print
        // println!("{:?} | {:?}: {}", self.idx, self.operation, out);
    }
}

// NOTE: This function is seperate to Node, as this avoids a borrowing problem
pub fn generate_deriv_nodes(
    operation: Operation,
    own_deriv_node: Idx,
    session: &mut Session,
) -> Option<Vec<(Idx, Idx)>> {
    // NOTE: Return Vec if same node is input twice
    // Own deriv Node = d_back / d_own
    // Generate partial derivs = d_own / d_in
    let partial_deriv = operation.gen_partial_derivs(session);

    if let Some(partial_deriv) = partial_deriv {
        let mul_partial_deriv: Vec<(Idx, Idx)> = partial_deriv
            .iter()
            .map(|(a, b)| {
                (
                    *a,
                    session.add_node(Operation::Multiply(own_deriv_node, *b)),
                )
            })
            .collect();
        return Some(mul_partial_deriv);
    }

    None
}
