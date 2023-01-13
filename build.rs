use std::env;
use std::path::PathBuf;
fn main() -> miette::Result<()> {
    // fn main() {
    println!("cargo:rerun-if-changed=tritonbackend.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("tritonbackend.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // let path = std::path::PathBuf::from("src"); // include path
    // let mut b = autocxx_build::Builder::new("src/lib.rs", &[&path])
    //     .build()
    //     .unwrap();
    // // This assumes all your C++ bindings are in lib.rs
    // b.flag_if_supported("-std=c++20")
    //     .file("src/minimal.cc")
    //     .compile("tburs"); // arbitrary library name, pick anything
    // println!("cargo:rerun-if-changed=src/lib.rs");
    // Add instructions to link to any C++ libraries you need.
    Ok(())
}
