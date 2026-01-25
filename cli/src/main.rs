#![feature(allocator_api)]

use std::{path::PathBuf, process::exit};

use clap::Parser;
use tsukiyotake::{
    Logos,
    engine::{
        CaptureSpan, LuaEngine, LuaError, ManagedValue, UnpackedValue, Value, Vec as TsuVec,
        table::Table,
    },
    hir::HirConversionContext,
    lex::Token,
    mir::{FunctionDef, MirConverter},
    parse::parse_block,
};

#[derive(Parser)]
#[command(version)]
struct Args {
    #[clap(long)]
    debug: bool,
    file: Option<PathBuf>,
    args: Option<Vec<String>>,
}

fn main() {
    let args = Args::parse();

    if let Some(file) = args.file {
        let input = std::fs::read_to_string(file).unwrap();

        let lexer = Token::lexer(&input);
        let ast = parse_block(&mut lexer.spanned().peekable()).unwrap();

        let conv = HirConversionContext::new();
        let hir = conv.convert_block(&ast);

        if args.debug {
            println!("--   HIR output   --");
            println!("{hir:#?}");
            println!("-- end HIR output --");
            println!();
        }

        let mut mir_conv = MirConverter::new_at_root();
        mir_conv.write_block(&hir);
        let mir = mir_conv.finish();

        if args.debug {
            println!("--   MIR output   --");
            println!("{mir:#?}");
            println!("-- end MIR output --");
            println!();
        }

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
    let mut os = Table::new(engine);
    os.insert(
        engine,
        Value::string_literal(b"exit"),
        engine.create_rust_function(os_exit),
    );
    table.insert(
        engine,
        Value::string_literal(b"os"),
        engine.allocate_managed_value(os),
    );
}

fn os_exit<'ctx>(
    _: &'ctx LuaEngine<'ctx>,
    params: &[Value<'ctx>],
) -> Result<TsuVec<'ctx, Value<'ctx>>, LuaError<'ctx>> {
    exit(params.get(0).and_then(|x| x.as_int()).unwrap_or(0) as i32)
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
        sep = " ";
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
        Ok(x) => {
            print(engine, x.as_slice())
                .map(|_| ())
                .unwrap_or_else(|e| eprintln!("{:?}", engine.debug_error(e)));
        }
        Err(e) => eprintln!("{:?}", engine.debug_error(e)),
    }
}
