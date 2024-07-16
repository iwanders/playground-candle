pub mod candle_util;
pub mod mnist;
// pub mod yolo;
pub mod fcn;
pub mod candle_fix;

pub type MainResult = anyhow::Result<()>;
pub fn main() -> MainResult {
        // mnist::network::main();
    /*
    if std::env::var("MNIST_MANUAL").is_ok() {
        mnist::manual::main()
    } else {
        mnist::network::main()
    }
    */
    // fcn::main()
    candle_fix::main()
}
