use std::any::Any;
use std::rc::Weak;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};
pub struct RenderGraph {
    heads: Vec<Rc<RefCell<RenderGraphHead>>>,
}

struct RenderGraphHead {
    nexts: Vec<Rc<RefCell<RenderGraphHead>>>,
    run: Box<dyn FnOnce()>,
}

pub struct Then {
    heads: Vec<Weak<RefCell<RenderGraphHead>>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self { heads: Vec::new() }
    }
    pub fn start<F: FnOnce() + 'static>(
        &mut self,
        run: F,
    ) -> Then {
        let node = Rc::new(RefCell::new(RenderGraphHead {
            nexts: Vec::new(),
            run: Box::new(run),
        }));
        let head = Rc::downgrade(&node);
        self.heads.push(node);
        Then {
            heads: vec![head],
        }
    }

    pub fn run(mut self) {
        while let Some(head) = self.heads.pop() {
            let mut head = match Rc::try_unwrap(head) {
                Ok(head) => head.into_inner(),
                Err(_) => panic!("Requries exclusive ownership"),
            };
            (head.run)();

            let new_heads = head.nexts.drain_filter(|next| Rc::strong_count(next) == 1);
            self.heads.extend(new_heads);
        }
    }
}

impl Then {
    pub fn then<F: FnOnce() + 'static>(
        &self,
        run: F,
    ) -> Then {
        let node = Rc::new(RefCell::new(RenderGraphHead {
            nexts: Vec::new(),
            run: Box::new(run),
        }));
        let head = Rc::downgrade(&node);
        for head in self.heads.iter() {
            head
            .upgrade()
            .unwrap() // This should still be referenced by the graph itself.
            .borrow_mut()
            .nexts
            .push(node.clone());
        }
        Then {
            heads: vec![head],
        }
    }
    pub fn join(&self, other: &Then) -> Then {
        let mut heads = Vec::with_capacity(self.heads.len() + other.heads.len());
        heads.extend_from_slice(&self.heads);
        heads.extend_from_slice(&other.heads);
        Then {
            heads,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        let mut graph = RenderGraph::new();
        let temp1 = graph
            .start(|| {
                println!("A");
            })
            .then(|| {
                println!("B");
            });

        let temp2 = temp1
            .then(|| {
                println!("C");
            })
            .then(|| {
                println!("D");
            });
        let temp3 = temp1
            .then(|| {
                println!("E");
            });

        let temp3 = temp1.join(&temp2).join(&temp3)
        .then(|| {
            println!("Success");
        });

        graph.run();
    }
}
