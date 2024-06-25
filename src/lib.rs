pub mod manual;

pub type MainResult = anyhow::Result<()>;
pub fn main() -> MainResult {
    manual::main()
}
