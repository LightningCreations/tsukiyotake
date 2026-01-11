#![feature(allocator_api)]

use std::path::PathBuf;

use clap::Parser;
use tsukiyotake::{
    engine::{
        CaptureSpan, LuaEngine, LuaError, ManagedValue, UnpackedValue, Value, Vec as TsuVec,
        table::Table,
    },
    grammar::BlockParser,
    hir::HirConversionContext,
    logos_lalrpop_bridge::Lexer,
    mir::{FunctionDef, MirConverter},
};

#[derive(Parser)]
#[command(version)]
struct Args {
    file: Option<PathBuf>,
    args: Option<Vec<String>>,
}

fn main() {
    let args = Args::parse();

    if let Some(file) = args.file {
        let input = std::fs::read_to_string(file).unwrap();

        let lexer = Lexer::new(&input);
        let parser = BlockParser::new();
        let ast = parser.parse(&input, lexer).unwrap();

        let conv = HirConversionContext::new();
        let hir = conv.convert_block(&ast);

        let mut mir_conv = MirConverter::new_at_root();
        mir_conv.write_block(&hir);
        let mir = mir_conv.finish();

        LuaEngine::with_userdata(65536, &mir, run_file, populate_env);
    } else {
        todo!("REPL");
    }
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
        print!(
            "{sep}{}",
            String::from_utf8_lossy(
                engine
                    .as_string(engine.do_tostring(*param).ok().unwrap())
                    .unwrap()
            )
        );
        sep = "\t";
    }
    println!();
    Ok(TsuVec::new_in(engine.alloc()))
}

fn run_file<'ctx>(engine: &'ctx LuaEngine<'ctx>, mir: &'ctx FunctionDef) {
    let func = engine.create_closure(&mir, CaptureSpan::Direct(TsuVec::new_in(engine.alloc())));
    let UnpackedValue::Managed(ManagedValue::Closure(func)) = func.unpack() else {
        unreachable!()
    };
    match engine.call_func(func, &[]) {
        Ok(_) => {}
        Err(e) => eprintln!("{:?}", engine.debug_error(e)),
    }
}
