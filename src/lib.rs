use rand::{thread_rng, Rng};
use rayon::prelude::*;

// Find the index of any value that matches. It will not nessisarily find the fist value.
fn which_c(dat: &[i32], find: i32) -> Option<usize> {
    // If the data is more than 100 elements long, use a parallel iterator for multi-core search
    // TODO: fine-tune this number to find where the cut off should be for multi-core search
    if dat.len() > 100 {
        dat.par_iter().position_any(|&r| r == find)
    } else {
        dat.iter().position(|&r| r == find)
    }
}

// Uses stochastic universal sampling algorithm to chose a random value between
// 1 and n using weights provided
pub fn sample_int<R: Rng>(weights: &[f64], n: usize, mut rng: R) -> usize {
    // TODO: We could skip this steps if we know weights sum to 1
    let sum: f64 = weights.iter().fold(0.0, |acc, &i| acc + i);
    let spoke_gap: f64 = sum / n as f64;

    // next_f64() ∈ [0.0, 1.0)
    let spin = rng.gen::<f64>() * spoke_gap;

    let mut i: usize = 0;
    let mut accumulated_weights = weights[0];

    while accumulated_weights < spin {
        i += 1;
        accumulated_weights += weights[i];
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find() {
        let items = vec![1, 2, 3, 4, 5];
        assert_eq!(which_c(&items, 1), Some(0));
        assert_eq!(which_c(&items, 2), Some(1));
        assert_eq!(which_c(&items, 5), Some(4));
        assert_eq!(which_c(&items, 6), None);
    }
}
