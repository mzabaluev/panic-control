extern crate panic_control;

mod prelude {
    // The extension trait under test.
    // No other types from the crate should be necessary to use.
    pub use panic_control::ThreadResultExt;

    pub use std::thread;
    pub use panic_control::spawn_quiet;

    pub const TEST_STR: &str = "catch me if you can!";
}

mod panic_value_as_str {
    use prelude::*;

    #[test]
    fn no_panic() {
        let h = thread::spawn(|| { });
        let res = h.join();
        assert!(res.is_ok());
        assert_eq!(res.panic_value_as_str(), None);
    }

    #[test]
    fn static_str_panic() {
        let h = spawn_quiet(|| {
            panic!(TEST_STR);
        });
        let res = h.join();
        assert!(res.is_err());
        assert_eq!(res.panic_value_as_str(), Some(TEST_STR));
    }

    #[test]
    fn string_panic() {
        let h = spawn_quiet(|| {
            panic!(String::from(TEST_STR));
        });
        let res = h.join();
        assert!(res.is_err());
        assert_eq!(res.panic_value_as_str(), Some(TEST_STR));
    }

    #[test]
    fn weird_panic() {
        let h = spawn_quiet(|| {
            panic!(42);
        });
        let res = h.join();
        assert!(res.is_err());
        assert_eq!(res.panic_value_as_str(), None);
    }
}

mod unwrap_or_propagate {
    use prelude::*;
    use panic_control::{Context, Outcome};

    #[test]
    fn no_panic() {
        let h = thread::spawn(|| { 42 });
        let res = h.join();
        assert_eq!(res.unwrap_or_propagate(), 42);
    }

    #[test]
    fn static_str_panic() {

        // Also demonstrating the reverse case here: when a common panic type
        // is expected, unexpected outcomes can be signalled by throwing custom
        // types.

        #[derive(Debug)]
        struct Unexpected(&'static str);

        let h = Context::<String>::new().spawn_quiet(|| {
            let h = spawn_quiet(|| {
                panic!(TEST_STR);
            });
            let res = h.join();
            res.unwrap_or_propagate();
            panic!(Unexpected("should have panicked above"));
        });
        let res = h.join();
        match res {
            Ok(Outcome::Panicked(s)) => {
                assert!(s.ends_with(TEST_STR));
            }
            Err(_) => panic!("unexpected panic occurred: {}",
                             res.panic_value_as_str()
                                .unwrap_or("{ .. }")),
            _ => panic!("unexpected result received: {:?}", res)
        }
    }

    #[test]
    fn string_panic() {
        let h = Context::<String>::new().spawn_quiet(|| {
            let h = spawn_quiet(|| {
                panic!(String::from(TEST_STR));
            });
            let res = h.join();
            res.unwrap_or_propagate();
        });
        let res = h.join();
        match res {
            Ok(Outcome::NoPanic(_)) => panic!("thread was expected to panic"),
            Ok(Outcome::Panicked(s)) => {
                assert!(s.ends_with(TEST_STR));
            }
            Err(_) => panic!("unexpected panic occurred: {}",
                             res.panic_value_as_str()
                                .unwrap_or("{ .. }"))
        }
    }

    #[test]
    fn weird_panic() {
        let h = Context::<&'static str>::new().spawn_quiet(|| {
            let h = spawn_quiet(|| {
                panic!(&[0]);
            });
            let res = h.join();
            res.unwrap_or_propagate();
        });
        let res = h.join();
        match res {
            Ok(Outcome::NoPanic(_)) => panic!("thread was expected to panic"),
            Ok(Outcome::Panicked(s)) => {
                assert!(s.ends_with("with undetermined value"));
            }
            Err(_) => panic!("unexpected panic occurred: {}",
                             res.panic_value_as_str()
                                .unwrap_or("{ .. }"))
        }
    }
}
