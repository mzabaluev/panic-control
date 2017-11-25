// Copyright 2017 Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// See the COPYRIGHT file at the top-level directory of this source tree.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::any::Any;
use std::cell::Cell;
use std::fmt;
use std::fmt::Debug;
use std::marker;
use std::panic;
use std::panic::{PanicInfo};
use std::thread;
use std::sync::{Once, ONCE_INIT};


#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome<T, P> {
    NoPanic(T),
    Panicked(P)
}

pub type ControlledPanicResult<T, P> = thread::Result<Outcome<T, P>>;

pub struct ControlledJoinHandle<T, P> {
    thread_handle: thread::JoinHandle<T>,
    phantom: marker::PhantomData<P>
}

impl<T, P> Debug for ControlledJoinHandle<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("ControlledJoinHandle { .. }")
    }
}

impl<T, P: Any> ControlledJoinHandle<T, P> {
    pub fn join(self) -> ControlledPanicResult<T, P> {
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

pub struct Context<P> {
    thread_builder: thread::Builder,
    phantom: marker::PhantomData<P>
}

impl<P> Debug for Context<P>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Context")
            .field("thread_builder", &self.thread_builder)
            .finish()
    }
}

impl<P: Any> Context<P> {
    pub fn new() -> Context<P> {
        Context {
            thread_builder: thread::Builder::new(),
            phantom: marker::PhantomData
        }
    }

    pub fn spawn<T, F>(self, f: F) -> ControlledJoinHandle<T, P>
        where F: FnOnce() -> T,
              F: Send + 'static,
              T: Send + 'static
    {
        let thread_handle = self.thread_builder.spawn(f).unwrap();
        ControlledJoinHandle {
            thread_handle: thread_handle,
            phantom: self.phantom
        }
    }

    pub fn spawn_quiet<T, F>(self, f: F) -> ControlledJoinHandle<T, P>
        where F: FnOnce() -> T,
              F: Send + 'static,
              T: Send + 'static
    {
        self.spawn(|| {
            disable_hook_in_current_thread();
            f()
        })
    }
}

impl<P: Any> From<thread::Builder> for Context<P> {
    fn from(builder: thread::Builder) -> Context<P> {
        Context {
            thread_builder: builder,
            phantom: marker::PhantomData
        }
    }
}

pub trait ThreadResultExt<T> {
    fn unwrap_or_propagate(self) -> T;
    fn panic_value_as_str(&self) -> Option<&str>;
}

fn str_from_any(something: &Any) -> Option<&str> {
    match something.downcast_ref::<&'static str>() {
        Some(s) => Some(*s),
        None => match something.downcast_ref::<String>() {
            Some(s) => Some(&s[..]),
            None    => None
        }
    }
}

fn propagate_panic(box_any: Box<Any + Send>) -> ! {
    match str_from_any(box_any.as_ref()) {
        Some(s) => panic!("observed an unexpected thread panic: {}", s),
        None => panic!("observed an unexpected thread panic \
                        with undetermined value")
    }
}

impl<T> ThreadResultExt<T> for thread::Result<T> {

    fn unwrap_or_propagate(self) -> T {
        match self {
            Ok(rv) => rv,
            Err(e) => propagate_panic(e)
        }
    }

    fn panic_value_as_str(&self) -> Option<&str> {
        match *self {
            Ok(_) => None,
            Err(ref ref_box_any) => str_from_any(&**ref_box_any)
        }
    }
}

pub fn chain_hook_ignoring<P: 'static>() {
    chain_hook_ignoring_if(|_: &P| { true })
}

pub fn chain_hook_ignoring_if<P, F>(predicate: F)
    where F: Fn(&P) -> bool,
          F: Send,
          F: Sync,
          F: 'static,
          P: 'static
{
    chain_hook_ignoring_full(move |info| {
        match info.payload().downcast_ref::<P>() {
            Some(p) => predicate(p),
            None => false
        }
    })
}

pub fn chain_hook_ignoring_full<F>(predicate: F)
    where F: Fn(&PanicInfo) -> bool,
          F: Send,
          F: Sync,
          F: 'static
{
    let next_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
            if !predicate(info) {
                next_hook(info);
            }
        }));
}

thread_local!(static IGNORE_HOOK: Cell<bool> = Cell::new(false));

fn init_thread_filter_hook() {
    static HOOK_ONCE: Once = ONCE_INIT;
    HOOK_ONCE.call_once(|| {
        chain_hook_ignoring_full(|_| {
            IGNORE_HOOK.with(|cell| { cell.get() })
        });
    });
}

pub fn disable_hook_in_current_thread() {
    init_thread_filter_hook();
    IGNORE_HOOK.with(|cell| {
        cell.set(true);
    });
}

pub fn enable_hook_in_current_thread() {
    init_thread_filter_hook();
    IGNORE_HOOK.with(|cell| {
        cell.set(false);
    });
}

pub fn spawn_quiet<T, F>(f: F) -> thread::JoinHandle<T>
    where F: FnOnce() -> T,
          F: Send + 'static,
          T: Send + 'static
{
    thread::spawn(|| {
        disable_hook_in_current_thread();
        f()
    })
}


#[cfg(test)]
mod tests {
    use super::{Context, Outcome};
    use super::{ThreadResultExt};
    use super::chain_hook_ignoring;
    use std::sync::{Once, ONCE_INIT};
    use std::thread;

    #[derive(Debug, PartialEq, Eq)]
    struct Expected(pub u32);

    fn silence_expected_panics() {
        static GUARD: Once = ONCE_INIT;
        GUARD.call_once(|| {
            chain_hook_ignoring::<Expected>()
        });
    }

    #[test]
    fn no_panic() {
        let ctx = Context::<Expected>::new();
        let h = ctx.spawn(|| {
            42
        });
        let outcome = h.join().unwrap();
        assert_eq!(outcome, Outcome::NoPanic(42));
    }

    #[test]
    // Have to help the compiler here: without an unreachable return value,
    // it has troubles inferring the no-panic type parameter of Outcome
    // and lints on Rust issue #39216.
    #[allow(unreachable_code)]
    fn expected_panic() {
        silence_expected_panics();
        let ctx = Context::<Expected>::new();
        let h = ctx.spawn(|| {
            panic!(Expected(42));
            ()
        });
        let outcome = h.join().unwrap();
        assert_eq!(outcome, Outcome::Panicked(Expected(42)));
    }

    #[test]
    fn int_literal_gotcha() {
        let h = Context::<u32>::new().spawn_quiet(|| {
            panic!(42);
        });
        // This wouldn't work:
        //     let outcome = h.join().unwrap();
        //     assert_eq!(outcome, Outcome::Panicked(42));
        let res = h.join();
        assert!(res.is_err());
    }

    #[test]
    fn from_thread_builder() {
        silence_expected_panics();
        const THREAD_NAME: &str = "a panicky thread";
        let thread_builder = thread::Builder::new()
                                .name(THREAD_NAME.into());
        let ctx = Context::<Expected>::from(thread_builder);
        let h = ctx.spawn(|| {
            let thread = thread::current();
            let name = thread.name();
            assert!(name.is_some());
            assert_eq!(name.unwrap(), THREAD_NAME);
            42
        });
        h.join().unwrap_or_propagate();
    }

    #[test]
    fn debug_impls_omit_phantom() {
        let ctx = Context::<Expected>::new();
        let repr = format!("{:?}", ctx);
        assert!(repr.starts_with("Context"));
        assert!(!repr.contains("phantom"));
        assert!(!repr.contains("PhantomData"));
        let h = ctx.spawn(|| { });
        let repr = format!("{:?}", h);
        assert_eq!(repr, "ControlledJoinHandle { .. }");
        h.join().unwrap();
    }
}
