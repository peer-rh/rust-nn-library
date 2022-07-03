use std::{collections::HashMap, hash::Hash};

use rand::{thread_rng, Rng};
use rust_autograd::{Graph, Idx, Session};

struct NeuralNetwork {
    session: Session,
    x_placeholders: Vec<Idx>,
    y_placeholders: Vec<Idx>,
    forward_graph: Graph,
    step_graph: Graph,
    error_idx: Idx,
    weights_values: HashMap<Idx, f32>,
    weights_gradients: HashMap<Idx, Idx>,
}

impl NeuralNetwork {
    fn new(layer_sizes: Vec<usize>) -> Self {
        let mut s = Session::new();
        let mut rng = thread_rng();
        let x_placeholders: Vec<Idx> = (0..layer_sizes[0]).map(|_| s.placeholder()).collect();
        let y_placeholders: Vec<Idx> = (0..layer_sizes[layer_sizes.len() - 1])
            .map(|_| s.placeholder())
            .collect();

        let mut all_weights: Vec<Idx> = Vec::new();
        let out_layer =
            layer_sizes[1..]
                .iter()
                .fold(x_placeholders.clone(), |prev_layer, this_size| {
                    let (mut weights, out) = generate_layer(&prev_layer, *this_size, &mut s);
                    all_weights.append(&mut weights);
                    out
                });

        println!("Out Len: {}", out_layer.len());

        let error = gen_error(&y_placeholders, &out_layer, &mut s);

        println!("Created All Nodes");
        let forward_graph = Graph::construct(out_layer, &s);
        println!("Generated Forward Graph");
        println!("Forward Graph has {} Nodes", forward_graph.get_n_nodes());
        let error_graph = Graph::construct(vec![error], &s);
        println!("Generated Error Graph");
        println!("Error Graph has {} Nodes", error_graph.get_n_nodes());
        let (deriv_graph, grad_nodes) = error_graph.generate_deriv_graph(&mut s);
        println!("Generated Grad Graph");
        println!("Grad Graph has {} Nodes", deriv_graph.get_n_nodes());

        let weights_values: HashMap<Idx, f32> = all_weights
            .iter()
            .map(|idx| (*idx, rng.gen::<f32>()))
            .collect();
        let weights_gradients: HashMap<Idx, Idx> = all_weights
            .iter()
            .map(|idx| (*idx, grad_nodes[idx]))
            .collect();

        let step_graph = {
            let mut out: Vec<Idx> = weights_gradients.values().cloned().collect();
            out.push(error);
            Graph::construct(out, &s)
        };

        NeuralNetwork {
            session: s,
            x_placeholders,
            y_placeholders,
            forward_graph,
            step_graph,
            error_idx: error,
            weights_values,
            weights_gradients,
        }
    }
    fn pred(&mut self, x: Vec<f32>) -> Vec<f32> {
        let mut feed_dict: HashMap<Idx, f32> = self
            .x_placeholders
            .iter()
            .zip(x.iter())
            .map(|(x_idx, x_val)| (*x_idx, *x_val))
            .collect();
        feed_dict.extend(self.weights_values.iter());
        self.session
            .eval_graph(&self.forward_graph, Some(feed_dict))
    }
    fn step(&mut self, x: Vec<f32>, y: Vec<f32>) -> f32 {
        let mut feed_dict: HashMap<Idx, f32> = self
            .x_placeholders
            .iter()
            .zip(x.iter())
            .map(|(x_idx, x_val)| (*x_idx, *x_val))
            .collect();

        let y_feed_dict = self
            .y_placeholders
            .iter()
            .zip(y.iter())
            .map(|(y_idx, y_val)| (*y_idx, *y_val));

        feed_dict.extend(y_feed_dict);
        feed_dict.extend(self.weights_values.iter());

        self.session.eval_graph(&self.step_graph, Some(feed_dict));

        for (w_idx, w_val) in self.weights_values.iter_mut() {
            *w_val -= self.session.get_value(&self.weights_gradients[w_idx]) * 0.01;
        }

        self.session.get_value(&self.error_idx)
    }
}

fn generate_layer(in_layer: &Vec<Idx>, out_size: usize, s: &mut Session) -> (Vec<Idx>, Vec<Idx>) {
    let mut all_weights: Vec<Idx> = Vec::with_capacity((in_layer.len() + 1) * out_size);
    let mut out_neurons: Vec<Idx> = Vec::with_capacity(out_size);
    for _ in 0..out_size {
        let (mut weights, out) = generate_neuron(in_layer, s);
        all_weights.append(&mut weights);
        out_neurons.push(out);
    }
    (all_weights, out_neurons)
}

fn generate_neuron(in_layer: &Vec<Idx>, s: &mut Session) -> (Vec<Idx>, Idx) {
    let weights: Vec<Idx> = (0..in_layer.len() + 1).map(|_| s.placeholder()).collect();
    let out: Idx = in_layer
        .iter()
        .zip(&weights[1..])
        .fold(weights[0], |neuron_value, (inp, w)| {
            let a = s.multiply(*inp, *w);
            s.sum(neuron_value, a)
        });
    (weights, out)
}

fn gen_error(y: &[Idx], y_pred: &[Idx], s: &mut Session) -> Idx {
    y.iter()
        .zip(y_pred.iter())
        .fold(s.constant(0.), |err, (this_y, this_y_pred)| {
            let a = s.subtract(*this_y, *this_y_pred);
            let b = s.square(a);
            s.sum(err, b)
        })
}

fn main() {
    let nn = NeuralNetwork::new(vec![784, 256, 128, 10]);
}
