pub mod manual;
pub mod network;
pub mod util;

pub type MainResult = anyhow::Result<()>;
pub fn main() -> MainResult {
    manual::main()
}
