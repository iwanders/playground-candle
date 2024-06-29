pub mod candle_util;
pub mod manual;
pub mod network;
pub mod util;

pub type MainResult = anyhow::Result<()>;
pub fn main() -> MainResult {
    if std::env::var("MNIST_MANUAL").is_ok() {
        manual::main()
    } else {
        network::main()
    }
}
