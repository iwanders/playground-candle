pub mod candle_util;
pub mod mnist;

pub type MainResult = anyhow::Result<()>;
pub fn main() -> MainResult {
    if std::env::var("MNIST_MANUAL").is_ok() {
        mnist::manual::main()
    } else {
        mnist::network::main()
    }
}
