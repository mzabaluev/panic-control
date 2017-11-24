// Copyright 2017 Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// See the COPYRIGHT file at the top-level directory of this source tree.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
pub struct JoinHandle<T, P> {
    thread_handle: thread::JoinHandle<T>,
    phantom: marker::PhantomData<P>
}

impl<T, P: Any> JoinHandle<T, P> {
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
}

pub fn spawn<T, P, F>(f: F) -> JoinHandle<T, P>
    where F: FnOnce() -> T,
          F: Send + 'static,
          T: Send + 'static,
          P: Any
{
    let thread_handle = thread::spawn(f);
    JoinHandle {
        thread_handle: thread_handle,
        phantom: marker::PhantomData
    }
}

fn report_unexpected_panic(info: &panic::PanicInfo) {
    use std::io;
    use std::io::Write;

    let thread = thread::current();
    let name = thread.name().unwrap_or("<unnamed>");
    let msg = if let Some(s) = info.payload().downcast_ref::<&'static str>() {
        *s
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        &s[..]
    } else {
        "Box<Any>"
    };
    match info.location() {
        Some(loc) => {
            let file = loc.file();
            let line = loc.line();
            let _ = writeln!(
                    io::stderr(),
                    "unexpected panic occurred in thread '{}' at '{}', {}:{}",
                    name, msg, file, line);
        }
        None => {
            let _ = writeln!(
                    io::stderr(),
                    "unexpected panic occurred in thread '{}' at '{}'",
                    name, msg);
        }
    }
}

pub fn set_hook_ignoring<P: 'static>() {
    panic::set_hook(Box::new(move |info| {
            if !info.payload().is::<P>() {
                report_unexpected_panic(info);
            }
        }));
}

pub use panic::take_hook;

#[cfg(test)]
mod tests {
    use super::spawn;
    use super::{JoinHandle, Outcome};
    use std::panic;

    fn shut_up_panic_hook() {
        panic::set_hook(Box::new(|_| { }));
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Expected(pub u32);

    #[test]
    fn expected_panic() {
        shut_up_panic_hook();
        let h: JoinHandle<(), Expected> = spawn(|| {
            panic!(Expected(42));
        });
        let outcome = h.join().unwrap();
        assert_eq!(outcome, Outcome::Panicked(Expected(42)));
    }

    #[test]
    fn int_literal_gotcha() {
        shut_up_panic_hook();
        let h: JoinHandle<(), u32> = spawn(|| {
            panic!(42);
        });
        // You probably assumed this would work:
        //     let outcome = h.join().unwrap();
        //     assert_eq!(outcome, Outcome::Panicked(42));
        let res = h.join();
        assert!(res.is_err());
    }
}
