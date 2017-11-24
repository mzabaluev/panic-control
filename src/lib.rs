// Copyright 2017 Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// See the COPYRIGHT file at the top-level directory of this source tree.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "panic_control"]
#![crate_type = "lib"]

use std::panic;
use std::thread;
use std::marker;
use std::any::Any;


#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome<T, P> {
    NoPanic(T),
    Panicked(P)
}

pub type PanicResult<T, P> = thread::Result<Outcome<T, P>>;

#[derive(Debug)]
pub struct Handle<T, P> {
    thread_handle: thread::JoinHandle<T>,
    phantom: marker::PhantomData<P>
}

impl<T, P: Any> Handle<T, P> {
    pub fn join(self) -> PanicResult<T, P> {
        match self.thread_handle.join() {
            Ok(rv) => Ok(Outcome::NoPanic(rv)),
            Err(box_any) => {
                match box_any.downcast::<P>() {
                    Ok(p)  => Ok(Outcome::Panicked(*p)),
                    Err(e) => Err(e)
                }
            }
        }
    }

    pub fn as_thread_join_handle(&self) -> &thread::JoinHandle<T> { &self.thread_handle }
    pub fn into_thread_join_handle(self) -> thread::JoinHandle<T> { self.thread_handle }
}

pub fn spawn<T, P, F>(f: F) -> Handle<T, P>
    where F: FnOnce() -> T,
          F: Send + 'static,
          T: Send + 'static,
          P: Any
{
    let thread_handle = thread::spawn(f);
    Handle {
        thread_handle: thread_handle,
        phantom: marker::PhantomData
    }
}

pub fn chain_hook_ignoring<P: 'static>() {
    let next_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
            if !info.payload().is::<P>() {
                next_hook(info);
            }
        }));
}

#[cfg(test)]
mod tests {
    use super::spawn;
    use super::{Handle, Outcome};
    use super::chain_hook_ignoring;
    use std::sync::{Once, ONCE_INIT};

    #[derive(Debug, PartialEq, Eq)]
    struct Expected(pub u32);

    fn ignore_expected_panics() {
        static GUARD: Once = ONCE_INIT;
        GUARD.call_once(|| {
            chain_hook_ignoring::<Expected>()
        });
    }

    #[test]
    fn expected_panic() {
        ignore_expected_panics();
        let h: Handle<(), Expected> = spawn(|| {
            panic!(Expected(42));
        });
        let outcome = h.join().unwrap();
        assert_eq!(outcome, Outcome::Panicked(Expected(42)));
    }

    #[test]
    fn int_literal_gotcha() {
        static GUARD: Once = ONCE_INIT;
        GUARD.call_once(|| {
            chain_hook_ignoring::<i32>();
        });

        let h: Handle<(), u32> = spawn(|| {
            panic!(42);
        });
        // This wouldn't work:
        //     let outcome = h.join().unwrap();
        //     assert_eq!(outcome, Outcome::Panicked(42));
        let res = h.join();
        assert!(res.is_err());
    }
}
