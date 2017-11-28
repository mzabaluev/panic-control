extern crate panic_control;

use panic_control::disable_hook_in_current_thread;
use panic_control::enable_hook_in_current_thread;
use panic_control::spawn_quiet;

use std::panic;
use std::thread;
use std::sync::mpsc;

// Don't create more tests in this file as we don't want the hook-setting
// code to run in parallel.
#[test]
fn thread_hook_tests() {

    let (sender, receiver) = mpsc::sync_channel::<String>(32);

    panic::set_hook(Box::new(move |info| {
        let payload = info.payload();
        if let Some(s) = payload.downcast_ref::<&'static str>() {
            let _ = sender.send(String::from(*s));
        } else if let Some(s) = payload.downcast_ref::<String>() {
            let _ = sender.send(s.clone());
        } else {
            let _ = sender.send("hook was invoked with an unknown value".into());
        }
    }));

    const TEST_STR: &str = "Panicked";

    // First, no thread filtering:
    let h = thread::spawn(|| {
        panic!(TEST_STR);
    });
    h.join().unwrap_err();
    let s = receiver.try_recv().expect("the panic hook was not invoked");
    assert_eq!(s, TEST_STR);

    let h = thread::spawn(|| {
        disable_hook_in_current_thread();
        panic!(());
    });
    h.join().unwrap_err();
    receiver.try_recv().expect_err("the panic hook was not disabled");

    let h = thread::spawn(|| {
        disable_hook_in_current_thread();
        enable_hook_in_current_thread();
        panic!(TEST_STR);
    });
    h.join().unwrap_err();
    let s = receiver.try_recv().expect("the panic hook was not enabled");
    assert_eq!(s, TEST_STR);

    let h = spawn_quiet(|| {
        panic!(());
    });
    h.join().unwrap_err();
    receiver.try_recv()
        .expect_err("spawn_quiet did not disable the panic hook");
}
