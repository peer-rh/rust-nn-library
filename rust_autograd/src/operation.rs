// TODO: Think about removing Node Struct

use std::collections::HashMap;

use super::{Idx, Node, Session};

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Constant(f32),
    // Variable(f32),
    Placeholder,
    Sum(Idx, Idx),
    Multiply(Idx, Idx),
    Negative(Idx),
    Exponential(Idx),
    Ln(Idx),
}

impl Operation {
    pub fn get_node(&self, idx: Idx) -> Node {
        Node::new(idx, *self)
    }

    pub fn forward(&self, current_values: &HashMap<Idx, f32>) -> f32 {
         match self {
            Self::Constant(a) => *a,
            // Self::Variable(a) => *a,
            Self::Sum(a, b) => current_values[a] + current_values[b],
            Self::Multiply(a, b) => current_values[a] * current_values[b],
            Self::Negative(a) => -current_values[a],
            Self::Exponential(a) => current_values[a].exp(),
            Self::Ln(a) => current_values[a].ln(),
            Self::Placeholder => 0., // Placeholder will not be called if initialized
        }
    }

    pub fn gen_partial_derivs(&self, session: &mut Session) -> Option<Vec<(Idx, Idx)>> {
        // NOTE: Returning Vec of tuple if the function has the same input twice
        match self {
            Self::Sum(a, b) => Some(vec![(*a, session.one()), (*b, session.one())]),
            Self::Multiply(a, b) => Some(vec![(*a, *b), (*b, *a)]),
            Self::Negative(a) => Some(vec![(*a, session.constant(-1.))]),
            Self::Ln(a) => {
                let one = session.one();
                Some(vec![(*a, session.divide(one, *a))])
            }
            Self::Exponential(a) => Some(vec![(*a, session.exp(*a))]),
            Self::Placeholder | Self::Constant(_) => None, // Placeholder will not be called if initialized
        }
    }

    pub fn get_input_nodes(&self) -> Option<Vec<Idx>> {
        match self {
            Self::Sum(a, b) | Self::Multiply(a, b) => Some(vec![*a, *b]),
            Self::Negative(a) | Self::Exponential(a) | Self::Ln(a) => Some(vec![*a]),
            _ => None,
        }
    }
}

impl Session {
    pub fn one(&mut self) -> Idx {
        self.constant(1.)
    }
    pub fn constant(&mut self, val: f32) -> Idx {
        self.add_node(Operation::Constant(val))
    }

    pub fn placeholder(&mut self) -> Idx {
        self.add_node(Operation::Placeholder)
    }

    pub fn sum(&mut self, a: Idx, b: Idx) -> Idx {
        self.add_node(Operation::Sum(a, b))
    }

    pub fn multiply(&mut self, a: Idx, b: Idx) -> Idx {
        self.add_node(Operation::Multiply(a, b))
    }

    pub fn square(&mut self, a: Idx) -> Idx {
        // NOTE: This is implemented to also be able to use negative numbers
        // Current Power requires complex numbers in the deriviative
        self.multiply(a, a)
    }

    pub fn divide(&mut self, a: Idx, b: Idx) -> Idx {
        let neg_one = self.constant(-1.);
        let b = self.pow(b, neg_one);
        self.multiply(a, b)
    }

    pub fn pow(&mut self, a: Idx, b: Idx) -> Idx {
        // Benchmark this against discrete implementation
        let c = self.ln(a);
        let d = self.multiply(c, b);
        self.exp(d)
    }

    pub fn exp(&mut self, a: Idx) -> Idx {
        self.add_node(Operation::Exponential(a))
    }

    pub fn ln(&mut self, a: Idx) -> Idx {
        self.add_node(Operation::Ln(a))
    }

    pub fn subtract(&mut self, a: Idx, b: Idx) -> Idx {
        let b = self.negative(b);
        self.sum(a, b)
    }

    pub fn negative(&mut self, a: Idx) -> Idx {
        self.add_node(Operation::Negative(a))
    }
}

#[cfg(test)]
mod tests {

    use crate::{Session, Graph};

    macro_rules! test_2_var {
        (
            $func_name:ident, 
            $var_1:expr, 
            $var_2:expr, 
            $operation:ident,
            $expected_out:expr,
            $expected_deriv_1:expr,
            $expected_deriv_2: expr
        ) => {
            #[test]
            fn $func_name() {
                let mut s = Session::new();
                let a = s.constant($var_1);
                let b = s.constant($var_2);
                let c = s.$operation(a, b);
                let graph = Graph::construct(vec![c], &s);
                let out = s.eval_graph(&graph, None);
                assert_eq!(out[0], $expected_out, "Expected Out doesnt equal");

                let (deriv_graph, deriv_nodes) = graph.generate_deriv_graph(&mut s);
                s.eval_graph(&deriv_graph, None);
                assert_eq!(s.get_value(&deriv_nodes[&a]), $expected_deriv_1, "Expected Deriv_1 doesn't equal");
                assert_eq!(s.get_value(&deriv_nodes[&b]), $expected_deriv_2, "Expected Deriv_1 doesn't equal");
            }
        };
    }

    macro_rules! test_1_var {
        (
            $func_name:ident, 
            $var:expr, 
            $operation:ident,
            $expected_out:expr,
            $expected_deriv:expr
        ) => {
            #[test]
            fn $func_name() {
                let mut s = Session::new();
                let a = s.constant($var);
                let c = s.$operation(a);
                let graph = Graph::construct(vec![c], &s);
                let out = s.eval_graph(&graph, None);
                assert_eq!(out[0], $expected_out, "Expected Out doesnt equal");

                let (deriv_graph, deriv_nodes) = graph.generate_deriv_graph(&mut s);
                s.eval_graph(&deriv_graph, None);
                assert_eq!(s.get_value(&deriv_nodes[&a]), $expected_deriv, "Expected Deriv doesn't equal");
            }
        };
    }

    test_2_var!(test_sum, 2., 3., sum, 5., 1., 1.);
    test_2_var!(test_multiply, 2., 3., multiply, 6., 3., 2.);
    test_2_var!(test_divide, 1., 2., divide, 0.5, 0.5, -0.25);
    test_2_var!(test_subtract, 2., 3., subtract, -1., 1., -1.);
    test_2_var!(test_power, 2., 3., pow, 8., 3. * 2_f32.powi(2),  2_f32.ln() * 2_f32.powi(3));

    test_1_var!(test_negative, 2., negative, -2., -1.);
    test_1_var!(test_exp, 2., exp, 2_f32.exp(), 2_f32.exp());
    test_1_var!(test_ln, 2., ln, 2_f32.ln(), 0.5);
    test_1_var!(test_square, 2., square, 4., 4.);
}
