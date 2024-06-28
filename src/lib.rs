pub mod candle_util;
pub mod manual;
pub mod network;
pub mod util;

pub type MainResult = anyhow::Result<()>;
pub fn main() -> MainResult {
    network::main()
    // manual::main()
}
