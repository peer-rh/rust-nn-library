use std::collections::HashMap;

use rust_autograd::{Graph, Session};

fn main() {
    let mut s = Session::new();
    let x = s.constant(2.);
    let w = s.placeholder();
    let y = s.constant(8.);
    let a = s.multiply(x, w);
    let error = s.subtract(a, y);
    let error = s.square(error);
    let graph = Graph::construct(vec![error], &s);
    let (deriv_graph, deriv_nodes) = graph.generate_deriv_graph(&mut s);
    let mut w_val = 3_f32;
    for _ in 0..100 {
        s.eval_graph(&graph, Some(HashMap::from([(w, w_val)])));
        s.eval_graph(&deriv_graph, None);
        let err = s.get_value(&error);
        let w_grad = s.get_value(&deriv_nodes[&w]);
        w_val -= w_grad * 0.01;

        println!("Error: {}, W_val: {}", err, w_val);
        s.reset();
    }
}
