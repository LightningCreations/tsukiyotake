#![feature(allocator_api)]

use tsukiyotake::engine::{LuaEngine, LuaError, Value, Vec as TsuVec, table::Table};

fn main() {
    LuaEngine::with(65536, run_app, populate_env);
}

fn populate_env<'ctx>(table: &mut Table<'ctx>, engine: &'ctx LuaEngine<'ctx>) {
    table.insert(
        engine,
        Value::string_literal(b"print"),
        engine.create_rust_function(print),
    );
}

fn print<'ctx>(
    engine: &'ctx LuaEngine<'ctx>,
    params: &[Value<'ctx>],
) -> Result<TsuVec<'ctx, Value<'ctx>>, LuaError<'ctx>> {
    let mut sep = "";
    for param in params {
        print!("{sep}{}", String::from_utf8_lossy(engine.as_string(*param).unwrap()));
        sep = "\t";
    }
    println!();
    Ok(TsuVec::new_in(engine.alloc()))
}

fn run_app<'ctx>(engine: &'ctx LuaEngine<'ctx>) {}
