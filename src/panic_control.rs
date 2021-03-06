// Copyright 2017 Mikhail Zabaluev <mikhail.zabaluev@gmail.com>
// See the COPYRIGHT file at the top-level directory of this source tree.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Controlled panics using dynamic type checking.
//!
//! Sometimes there is a need to test how Rust code behaves on occurrence
//! of a panic. A panic can be invoked on purpose in a thread spawned by the
//! test and the effects observed after the thread is joined.
//! The problem with "benign" panics is that it may be cumbersome to tell them
//! apart from panics indicating actual errors, such as assertion failures.
//!
//! Another issue is the behavior of the default panic hook.
//! It is very useful for getting information about the cause of an
//! unexpected thread panic, but for tests causing panics on purpose it
//! produces annoying output noise. The panic hook can be overridden,
//! but custom panic hooks affect the entire program, which in typical
//! usage is the test runner; it is easy to misuse them causing important
//! error information to go unreported.
//!
//! The simplest way, as provided by the standard library, to propagate
//! a panic that occurred in a child thread to the thread that spawned it
//! is to call `unwrap` on the result of `JoinHandle::join`. Unfortunately,
//! due to [an issue](https://github.com/rust-lang/rfcs/issues/1389) with
//! the implementation of `Any`, the resulting panic message does not relay
//! information from the child thread's panic.
//!
//! This crate provides utilities and an ergonomic interface for testing
//! panics in a controlled and output-friendly way using dynamic type checks
//! to discern between expected and unexpected panics.
//!
//! # Expected Panic Type
//!
//! The recommended way to designate panics as expected is by using values of
//! a custom type as the parameter for `panic!`. The type could be as simple
//! as a token unit-like struct, or it can be equipped to carry additional
//! information from the panic site.
//! Any panic value type shall be `Sized`, `'static`, and `Send`.
//! For the value to be usable in testing, it should also implement
//! at least `Debug` and `PartialEq`.
//!
//! # Examples
//!
//! ```
//! use panic_control::{Context, Outcome};
//! use panic_control::{chain_hook_ignoring, spawn_quiet};
//! use panic_control::ThreadResultExt;
//!
//! use std::thread;
//!
//! #[derive(Debug, PartialEq, Eq)]
//! enum Expected {
//!     Token,
//!     Int(i32),
//!     String(String)
//! }
//!
//! // Rust's stock test runner does not provide a way to do global
//! // initialization and the tests are run in parallel in a random
//! // order by default. So this is our solution, to be called at
//! // the beginning of every test exercising a panic with an
//! // Expected value.
//! fn silence_expected_panics() {
//!     use std::sync::{Once, ONCE_INIT};
//!     static HOOK_ONCE: Once = ONCE_INIT;
//!     HOOK_ONCE.call_once(|| {
//!         chain_hook_ignoring::<Expected>()
//!     });
//! }
//!
//! # struct TypeUnderTest;
//! # impl TypeUnderTest {
//! #     fn new() -> TypeUnderTest { TypeUnderTest }
//! #     fn doing_fine(&self) -> bool { true }
//! # }
//! // ...
//!
//! silence_expected_panics();
//! let thread_builder = thread::Builder::new()
//!                      .name("My panicky thread".into());
//! let ctx = Context::<Expected>::from(thread_builder);
//! let h = ctx.spawn(|| {
//!     let unwind_me = TypeUnderTest::new();
//!     assert!(unwind_me.doing_fine());
//!          // ^-- If this fails, join() will return Err
//!     panic!(Expected::String("Rainbows and unicorns!".into()));
//! });
//! let outcome = h.join().unwrap_or_propagate();
//! match outcome {
//!     Outcome::Panicked(Expected::String(s)) => {
//!         println!("thread panicked as expected: {}", s);
//!     }
//!     _ => panic!("unexpected value returned from join()")
//! }
//!
//! let ctx = Context::<Expected>::new();
//! let h = ctx.spawn_quiet(|| {
//!     let h = spawn_quiet(|| {
//!         panic!("Sup dawg, we heard you like panics \
//!                 so we put a panic in your panic!");
//!     });
//!     h.join().unwrap_or_propagate();
//! });
//! let res = h.join();
//! let msg = res.panic_value_as_str().unwrap();
//! assert!(msg.contains("panic in your panic"));
//! ```

use std::any::Any;
use std::cell::Cell;
use std::fmt;
use std::fmt::Debug;
use std::marker;
use std::panic;
use std::panic::{PanicInfo};
use std::thread;
use std::sync::{Once, ONCE_INIT};


/// Enumerates the expected outcomes from joining a panic-checked thread.
///
/// `Outcome` values are returned in the `Ok` result variant
/// of the `join()` method of a `CheckedJoinHandle`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome<T, P> {
    /// Indicates that the thread closure has
    /// returned normally, and provides the return value.
    NoPanic(T),
    /// Indicates that the thread has panicked with the expected type,
    /// and provides the panic value.
    Panicked(P)
}

impl<T, P> Outcome<T, P> {
    /// Tests whether the value contains the variant `Panicked`.
    pub fn has_panicked(&self) -> bool {
        match *self {
            Outcome::NoPanic(_) => false,
            Outcome::Panicked(_) => true,
        }
    }
}

/// Wraps `std::thread::JoinHandle` for panic value discrimination.
///
/// A `CheckedJoinHandle` works like a standard `JoinHandle`,
/// except that its `join()` method dynamically checks the type of
/// the possible panic value for matching the type that is the
/// parameter of the `Context` this handle was obtained from,
/// and if the type matches, returns the resolved value in the
/// "successful panic" result variant.
///
/// See the documentation of the `join()` method for details and
/// an example of use.
pub struct CheckedJoinHandle<T, P> {
    thread_handle: thread::JoinHandle<T>,
    phantom: marker::PhantomData<P>
}

impl<T, P> Debug for CheckedJoinHandle<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("CheckedJoinHandle { .. }")
    }
}

impl<T, P: Any> CheckedJoinHandle<T, P> {

    /// Works like `std::thread::JoinHandle::join()`, except that when
    /// the child thread's panic value is of the expected type, it is
    /// returned in `Ok(Outcome::Panicked(_))`. If the child thread's
    /// closure returns normally, its return value is returned in
    /// `Ok(Outcome::NoPanic(_))`
    ///
    /// # Examples
    ///
    /// ```
    /// use panic_control::{Context, Outcome};
    /// use std::thread;
    ///
    /// #[derive(Debug, PartialEq, Eq)]
    /// struct Expected(pub u32);
    ///
    /// let ctx = Context::<Expected>::new();
    /// let h = ctx.spawn(|| {
    ///     panic!(Expected(42));
    /// });
    ///
    /// let outcome = h.join().unwrap();
    ///
    /// match outcome {
    ///     Outcome::Panicked(Expected(n)) => {
    ///         println!("thread panicked as expected with {}", n);
    ///     }
    ///     _ => panic!("unexpected return value from join()")
    /// }
    /// ```
    pub fn join(self) -> thread::Result<Outcome<T, P>> {
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

impl<T, P> CheckedJoinHandle<T, P> {

    /// Returns a reference to the underlying `JoinHandle`.
    pub fn as_thread_join_handle(&self) -> &thread::JoinHandle<T> {
        &self.thread_handle
    }

    /// Converts into the underlying `JoinHandle`,
    /// giving up panic discrimination.
    pub fn into_thread_join_handle(self) -> thread::JoinHandle<T> {
        self.thread_handle
    }
}

impl<T, P> AsRef<thread::JoinHandle<T>> for CheckedJoinHandle<T, P> {
    fn as_ref(&self) -> &thread::JoinHandle<T> {
        self.as_thread_join_handle()
    }
}

impl<T, P> Into<thread::JoinHandle<T>> for CheckedJoinHandle<T, P> {
    fn into(self) -> thread::JoinHandle<T> {
        self.into_thread_join_handle()
    }
}

/// The launch pad for a thread checked for the expected panic type.
///
/// The generic type `Context` serves as the type system's anchor for the
/// expected type of the panic value, which is given as the type parameter.
/// It can be constructed from an `std::thread::Builder` providing
/// a customized thread configuration.
///
/// The method `spawn()` is used to spawn a new thread, similarly to
/// how the function `std::thread::spawn()` works. See the documentation
/// of the `spawn()` method for detailed description and examples.
///
/// # Examples
///
/// The example below demonstrates how to construct a `Context` from a
/// configured `std::thread::Builder` using the implementation of
/// the standard trait `From`.
///
/// ```
/// use panic_control::Context;
/// use std::thread;
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct ExpectedToken;
///
/// let thread_builder = thread::Builder::new()
///                      .name("My panicky thread".into())
///                      .stack_size(65 * 1024);
/// let ctx = Context::<ExpectedToken>::from(thread_builder);
/// let h = ctx.spawn(|| {
///     // ...
/// });
/// h.join().unwrap();
/// ```
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

impl<P: Any> Default for Context<P> {
    fn default() -> Context<P> { Context::new() }
}

impl<P: Any> Context<P> {

    /// Constructs a context with the default thread configuration.
    ///
    /// The type parameter can be specified explicitly:
    ///
    /// ```
    /// # use panic_control::Context;
    /// # #[derive(Debug, PartialEq)] struct Expected(pub i32);
    /// let ctx = Context::<Expected>::new();
    /// ```
    pub fn new() -> Context<P> {
        Context {
            thread_builder: thread::Builder::new(),
            phantom: marker::PhantomData
        }
    }

    /// Spawns a new thread taking ownership of the `Context`, and
    /// returns the `CheckedJoinHandle` for the thread. Other than the
    /// return value, and panicking on an OS failure to create a thread,
    /// this method behaves exactly like the `spawn()`
    /// method of `std::thread::Builder` does, and if the `Context`
    /// was constructed from a `std::thread::Builder`, its
    /// thread configuration will be applied.
    ///
    /// # Panics
    ///
    /// Panics if the underlying call to `std::thread::Builder::spawn()`
    /// returns an `Err` value. Any panics that function can cause apply
    /// as well.
    ///
    /// # Examples
    ///
    /// The example below uses some kludges to work around a compiler
    /// quirk:
    ///
    /// ```
    /// # use panic_control::{Context, Outcome};
    /// # #[derive(Debug, PartialEq)] struct Expected(pub i32);
    /// let ctx = Context::<Expected>::new();
    ///
    /// #[allow(unreachable_code)]
    /// let h = ctx.spawn(|| {
    ///     panic!(Expected(42));
    ///     ()
    /// });
    ///
    /// let outcome = h.join().unwrap();
    /// assert_eq!(outcome, Outcome::Panicked(Expected(42)));
    /// ```
    ///
    /// Note that unless the unreachable return expression is present,
    /// the compiler would have no information to infer the unspecified
    /// first type parameter of `Outcome`, so it would settle on a default type
    /// that is going to be changed to `!` in Rust 1.26. In previous versions
    /// of the compiler, code that invokes this behavior is
    /// [denied by a lint](https://github.com/rust-lang/rust/issues/39216),
    /// so the following example does not compile:
    ///
    /// ```ignore
    /// # use panic_control::{Context, Outcome};
    /// # #[derive(Debug, PartialEq)] struct Expected(pub i32);
    /// let ctx = Context::<Expected>::new();
    /// let h = ctx.spawn(|| {
    ///     panic!(Expected(42));
    /// });
    /// let outcome = h.join().unwrap();
    /// assert_eq!(outcome, Outcome::Panicked(Expected(42)));
    /// ```
    ///
    /// A way to avoid the future incompatibility without resorting
    /// to warning overrides is to match the `Outcome` value without
    /// touching the sensitive parts:
    ///
    /// ```
    /// # use panic_control::{Context, Outcome};
    /// # #[derive(Debug, PartialEq)] struct Expected(pub i32);
    /// let ctx = Context::<Expected>::new();
    /// let h = ctx.spawn(|| {
    ///     panic!(Expected(42));
    /// });
    /// let outcome = h.join().unwrap();
    /// match outcome {
    ///     Outcome::Panicked(Expected(n)) => assert_eq!(n, 42),
    ///     _ => panic!("join() returned an unexpected Outcome value")
    /// }
    /// ```
    pub fn spawn<T, F>(self, f: F) -> CheckedJoinHandle<T, P>
        where F: FnOnce() -> T,
              F: Send + 'static,
              T: Send + 'static
    {
        let thread_handle = self.thread_builder.spawn(f).unwrap();
        CheckedJoinHandle {
            thread_handle: thread_handle,
            phantom: self.phantom
        }
    }

    /// Like `spawn()`, but disables the panic hook
    /// if the spawned thread panics.
    ///
    /// The process-global panic hook, either installed with
    /// `std::panic::set_hook()` or the standard library default,
    /// gets augmented with a filter that disables invocation of the
    /// hook closure if the spawned thread panics.
    ///
    /// This function can be used in any order together with other
    /// functions and methods of this crate that modify the panic hook.
    ///
    /// # Caveats
    ///
    /// Note that the suppression can apply to the default panic hook
    /// that is normally used to report assertion failures and other
    /// unexpected panics on the standard error stream.
    /// The only remaining way to observe the panic is by checking
    /// the result of `join()` for the spawned thread.
    ///
    /// Other code within the program that modifies the panic hook,
    /// concurrently to, or after, a call to this function, may cause
    /// the suppression not to work as intended. See the documentation
    /// on the function `disable_hook_in_current_thread()` for possible
    /// pitfalls.
    ///
    /// # Examples
    ///
    /// ```
    /// # use panic_control::{Context, Outcome};
    /// # #[derive(Debug, PartialEq)] struct Expected(pub i32);
    /// let ctx = Context::<Expected>::new();
    /// let h = ctx.spawn_quiet(|| {
    ///     assert!(false, "I'm panicking, \
    ///         but you can only learn about it through join()");
    /// });
    /// let res = h.join();
    /// assert!(res.is_err());
    /// ```
    pub fn spawn_quiet<T, F>(self, f: F) -> CheckedJoinHandle<T, P>
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

/// Helpful extension methods for `std::thread::Result`.
///
/// The `Result` type defined in `std::thread` is a
/// specialization of the standard `Result` with a `Box<Any>`
/// in the `Err` variant, which receives the payload
/// value of a panic.
/// As such, `Result` does not provide convenient ways
/// to examine the content of the panic value. Furthermore,
/// the generic implementations of `unwrap()` and related methods
/// use the `Debug` implementation of the content of `Err` to format
/// the panic message, which is
/// [not very useful](https://github.com/rust-lang/rfcs/issues/1389)
/// in case of `Any`.
///
/// When this trait is used in a lexical scope, it augments
/// any `Result` value that matches the specialization of
/// `std::thread::Result` with methods that facilitate
/// examination and reporting of the possible string value
/// which is most often found in the dynamically typed
/// `Err` variant. The methods are meant to be used on the
/// result of `std::thread::JoinHandle::join()` or
/// `CheckedJoinHandle::join()`.
///
/// # Examples
///
/// ```
/// use panic_control::ThreadResultExt;
///
/// use panic_control::Context;
/// use std::thread;
///
/// let h = thread::spawn(|| {
///     42
/// });
/// let n = h.join().unwrap_or_propagate();
/// assert_eq!(n, 42);
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct Expected;
///
/// let ctx = Context::<Expected>::new();
/// let h = ctx.spawn_quiet(|| {
///     panic!("No coffee beans left in the bag!");
/// });
/// let res = h.join();
/// let msg = res.panic_value_as_str().unwrap();
/// println!("{}", msg);
/// ```
///
pub trait ThreadResultExt<T> : sealed::ThreadResultExtSealed {

    /// Unwraps a result, yielding the content of an `Ok`.
    ///
    /// # Panics
    ///
    /// Panics if the value is an `Err`, with a panic message appended with
    /// the `Err`'s string value if that is found to be such, or a generic
    /// message otherwise. The message is meant to relay information from
    /// a panic in a child thread that is observed through this result value,
    /// as returned by `std::thread::JoinHandle::join()` or
    /// `CheckedJoinHandle::join()`.
    fn unwrap_or_propagate(self) -> T;

    /// If the value is an `Err` and its content is a string, returns the
    /// content as a string slice. Otherwise returns `None`.
    fn panic_value_as_str(&self) -> Option<&str>;
}

fn str_from_any(something: &Any) -> Option<&str> {
    if let Some(s) = something.downcast_ref::<&'static str>() {
        Some(*s)
    } else if let Some(s) = something.downcast_ref::<String>() {
        Some(&s[..])
    } else {
        None
    }
}

fn propagate_panic(box_any: Box<Any + Send>) -> ! {
    match str_from_any(box_any.as_ref()) {
        Some(s) => panic!("observed an unexpected thread panic: {}", s),
        None => panic!("observed an unexpected thread panic \
                        with undetermined value")
    }
}

mod sealed {
    use std::thread;

    // trait ThreadResultExt is sealed by this crate: it should only
    // make sense for thread::Result.

    pub trait ThreadResultExtSealed { }
    impl<T> ThreadResultExtSealed for thread::Result<T> { }
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

/// Augments the panic hook, filtering out panics of a particular type.
///
/// The current panic hook, either installed with `std::panic::set_hook()`
/// or the standard library default, gets chained, again using
/// `std::panic::set_hook()`, behind
/// a dynamic type check for the panic payload. If it is found to be
/// of the same type as the type parameter of this generic function,
/// the chained hook is not called.
///
/// # Caveats
///
/// Every call to this function allocates state data and increases the
/// filtering chain for the process-global panic hook, so it should not be
/// called repeatedly unless necessary.
/// Other code within the program that modifies the panic hook, concurrently
/// to, or after, the call to this function, may cause the hook chain to stop
/// working as intended.
/// This function interoperates in a predictable way only with the other
/// functions and methods of this crate that modify the panic hook,
/// and only when used in strictly serialized order with those functions,
/// unless noted otherwise in those functions' documentation.
/// This function is only intended to be used in tests or initialization
/// code of a program;
/// libraries other than those designed for test purposes should avoid
/// using it.
///
/// # Examples
///
/// ```
/// use panic_control::chain_hook_ignoring;
/// use std::sync::{Once, ONCE_INIT};
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct PanicToken;
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct PanicMessage(pub String);
///
/// static HOOK_ONCE: Once = ONCE_INIT;
/// HOOK_ONCE.call_once(|| {
///     chain_hook_ignoring::<PanicToken>();
///     chain_hook_ignoring::<PanicMessage>();
/// });
/// ```
pub fn chain_hook_ignoring<P: 'static>() {
    chain_hook_ignoring_if(|_: &P| { true })
}

/// Augments the panic hook, filtering out panics with a
/// statically typed closure.
///
/// The current panic hook, either installed with `std::panic::set_hook()`
/// or the standard library default, gets chained, again using
/// `std::panic::set_hook()`, behind
/// the boolean predicate closure passed as the parameter, testing a value
/// of a particular type.
/// If the panic payload is found to be of the same type and the predicate
/// returns true, the chained hook is not called.
///
/// # Caveats
///
/// Every call to this function allocates state data and increases the
/// filtering chain for the process-global panic hook, so it should not be
/// called repeatedly unless necessary.
/// Other code within the program that modifies the panic hook, concurrently
/// to, or after, the call to this function, may cause the hook chain to stop
/// working as intended.
/// This function interoperates in a predictable way only with the other
/// functions and methods of this crate that modify the panic hook,
/// and only when used in strictly serialized order with those functions,
/// unless noted otherwise in those functions' documentation.
/// This function is only intended to be used in tests or initialization
/// code of a program;
/// libraries other than those designed for test purposes should avoid
/// using it.
///
/// # Examples
///
/// The value types most commonly used in panics are `&'string str` or
/// `String`, depending on whether the `panic!` macro was used in the single
/// parameter form or the formatting form, respectively. The example below
/// filters out either kind if the string message contains a particular
/// substring.
///
/// ```
/// use panic_control::chain_hook_ignoring_if;
/// use std::sync::{Once, ONCE_INIT};
///
/// const MAGIC: &str = "Move along, nothing to see here";
///
/// static HOOK_ONCE: Once = ONCE_INIT;
/// HOOK_ONCE.call_once(|| {
///     chain_hook_ignoring_if(|s: &&'static str| {
///         s.contains(MAGIC)
///     });
///     chain_hook_ignoring_if(|s: &String| {
///         s.contains(MAGIC)
///     });
/// });
/// ```
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

/// Augments the panic hook, filtering out panics with a free-form check.
///
/// The current panic hook, either installed with `std::panic::set_hook()`
/// or the standard library default, gets chained, again using
/// `std::panic::set_hook()`, behind
/// the boolean predicate closure passed as the parameter, testing the
/// `std::panic::PanicInfo` structure passed to the panic hook.
/// If the predicate returns true, the chained hook is not called.
///
/// # Caveats
///
/// Every call to this function allocates state data and increases the
/// filtering chain for the process-global panic hook, so it should not be
/// called repeatedly unless necessary.
/// Other code within the program that modifies the panic hook, concurrently
/// to, or after, the call to this function, may cause the hook chain to stop
/// working as intended.
/// This function interoperates in a predictable way only with the other
/// functions and methods of this crate that modify the panic hook,
/// and only when used in strictly serialized order with those functions,
/// unless noted otherwise in those functions' documentation.
/// This function is only intended to be used in tests or initialization
/// code of a program;
/// libraries other than those designed for test purposes should avoid
/// using it.
///
/// # Examples
///
/// This example filters out any non-string panics:
///
/// ```
/// use panic_control::chain_hook_ignoring_full;
/// use std::sync::{Once, ONCE_INIT};
///
/// static HOOK_ONCE: Once = ONCE_INIT;
/// HOOK_ONCE.call_once(|| {
///     chain_hook_ignoring_full(|info| {
///         let payload = info.payload();
///         !(payload.is::<&'static str>() ||
///           payload.is::<String>())
///     });
/// });
/// ```
pub fn chain_hook_ignoring_full<F>(predicate: F)
    where F: Fn(&PanicInfo) -> bool,
          F: Send,
          F: Sync,
          F: 'static
{
    // Make sure the thread filter hook is set up,
    // in case some other thread is calling
    // {disable,enable}_hook_in_current_thread()
    init_thread_filter_hook();
    chain_hook_waive_init_thread_filter(predicate)
}

fn chain_hook_waive_init_thread_filter<F>(predicate: F)
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
        // Avoid recursion and deadlocking on HOOK_ONCE here
        chain_hook_waive_init_thread_filter(|_| {
            IGNORE_HOOK.with(|cell| { cell.get() })
        });
    });
}

/// Disables the panic hook for the current thread.
///
/// The process-global panic hook, either installed with
/// `std::panic::set_hook()` or the standard library default,
/// gets augmented with a filter that disables invocation of the
/// hook closure if the thread that is calling this function panics.
///
/// This function does not allocate resources when called repeatedly in
/// the same thread, and it can be used in any order together with other
/// functions and methods of this crate that modify the panic hook.
///
/// # Caveats
///
/// Note that the suppression can apply to the default panic hook
/// that is normally used to report assertion failures and other
/// unexpected panics on the standard error stream.
///
/// Other code within the program that modifies the panic hook, concurrently
/// to, or after, a call to this function, may cause the hook chain to stop
/// working as intended.
/// This function interoperates in a predictable way only with the other
/// functions and methods of this crate that modify the panic hook.
/// Libraries other than those designed for test purposes should avoid
/// using this function.
pub fn disable_hook_in_current_thread() {
    init_thread_filter_hook();
    IGNORE_HOOK.with(|cell| {
        cell.set(true);
    });
}

/// Enables the panic hook for the current thread.
///
/// If the panic hook has been disabled for the current thread with
/// `disable_hook_in_current_thread()`, calling this function enables it
/// back.
///
/// This function does not allocate resources when called repeatedly in
/// the same thread, and it can be used in any order together with other
/// functions and methods of this crate that modify the panic hook.
///
/// # Caveats
///
/// Other code within the program that modifies the panic hook, concurrently
/// to, or after, a call to this function, may cause the hook chain to stop
/// working as intended.
/// This function interoperates only with the other
/// functions and methods of this crate that modify the panic hook.
/// Libraries other than those designed for test purposes should avoid
/// using this function.
pub fn enable_hook_in_current_thread() {
    init_thread_filter_hook();
    IGNORE_HOOK.with(|cell| {
        cell.set(false);
    });
}

/// Like `std::thread::spawn()`, but disables the panic hook
/// if the spawned thread panics.
///
/// The process-global panic hook, either installed with
/// `std::panic::set_hook()` or the standard library default,
/// gets augmented with a filter that disables invocation of the
/// hook closure if the spawned thread panics.
///
/// This function can be used in any order together with other
/// functions and methods of this crate that modify the panic hook.
///
/// # Caveats
///
/// Note that the suppression can apply to the default panic hook
/// that is normally used to report assertion failures and other
/// unexpected panics on the standard error stream.
/// The only remaining way to observe the panic is by checking
/// the result of `join()` for the spawned thread.
///
/// Other code within the program that modifies the panic hook,
/// concurrently to, or after, a call to this function, may cause
/// the suppression not to work as intended. See the documentation
/// on the function `disable_hook_in_current_thread()` for possible
/// pitfalls.
///
/// # Examples
///
/// ```
/// use panic_control::spawn_quiet;
///
/// let h = spawn_quiet(|| {
///     assert!(false, "I'm panicking, \
///         but you can only learn about it through join()");
/// });
/// let res = h.join();
/// assert!(res.is_err());
/// ```
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
    fn context_default() {
        let ctx = Context::<Expected>::default();
        // Also use spawn_quiet and exercise the normal return path there
        let h = ctx.spawn_quiet(|| {
            42
        });
        let outcome = h.join().unwrap();
        assert_eq!(outcome, Outcome::NoPanic(42));
    }

    #[test]
    fn expected_panic() {
        silence_expected_panics();
        let ctx = Context::<Expected>::new();
        let h = ctx.spawn(|| {
            panic!(Expected(42));
        });
        let outcome = h.join().unwrap();
        match outcome {
            Outcome::Panicked(Expected(n)) => assert_eq!(n, 42),
            _ => panic!("unexpected Outcome value returned from join()")
        }
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
    fn checked_join_handle_inherent_as_ref() {
        const THREAD_NAME: &str = "a non-panicky thread";
        let thread_builder = thread::Builder::new()
                                .name(THREAD_NAME.into());
        let ctx = Context::<Expected>::from(thread_builder);
        let h = ctx.spawn(|| {});
        {
            let h = h.as_thread_join_handle();
            let name = h.thread().name().unwrap();
            assert_eq!(name, THREAD_NAME);
        }
        h.join().unwrap_or_propagate();
    }

    #[test]
    fn checked_join_handle_trait_as_ref() {
        const THREAD_NAME: &str = "a non-panicky thread";
        let thread_builder = thread::Builder::new()
                                .name(THREAD_NAME.into());
        let ctx = Context::<Expected>::from(thread_builder);
        let h = ctx.spawn(|| {});
        {
            let h: &thread::JoinHandle<()> = h.as_ref();
            let name = h.thread().name().unwrap();
            assert_eq!(name, THREAD_NAME);
        }
        h.join().unwrap_or_propagate();
    }

    #[test]
    fn checked_join_handle_inherent_into() {
        silence_expected_panics();
        let ctx = Context::<Expected>::new();
        let h = ctx.spawn(|| {
            panic!(Expected(42));
        });
        let h = h.into_thread_join_handle();
        let res = h.join();
        assert!(res.is_err());
    }

    #[test]
    fn checked_join_handle_trait_into() {
        silence_expected_panics();
        let ctx = Context::<Expected>::new();
        #[allow(unreachable_code)]
        let h = ctx.spawn(|| {
            panic!(Expected(42));
            ()
        });
        let h: thread::JoinHandle<()> = h.into();
        let res = h.join();
        assert!(res.is_err());
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
        assert_eq!(repr, "CheckedJoinHandle { .. }");
        h.join().unwrap();
    }
}
