use jlrs::{
    convert::unbox::Unbox,
    prelude::*,
};

use crate::triton::backend::julia::jb_tensor::JbTensor;

julia_module! {
    become triton_backend_jl_init;
    struct JbTensor as Tensor;

}

/* 
julia_module! {
     // init_function_name is the name of the generated initialization function.
     //
     // The name of the generated function must be unique, it's recommended you prefix it with
     // the crate name. If your crate is named foo-jl, you should use a name like
     // `foo_jl_init`.
     become init_function_name;

     // Exports the function `foo` as `bar` with documentation.
     //
     // The `unsafe extern "C" part of the signature must be elided, the signature is verified
     // in the generated code to ensure it's correct and that the function uses the C ABI.
     //
     // The `as <exposed_name>` part is optional, by default the function is exported with the
     // name it has in Rust, the exposed name can end in an exclamation mark.
     //
     // A docstring can be provided with the doc attribute; if multiple functions are exported
     // with the same name it shoud only be documented once. All exported items can be
     // documented, a multi-line docstring can be created by providing multiple doc attributes
     // for the same item.
     #[doc = "    bar(arr::Array)"]
     #[doc = ""]
     #[doc = "Documentation for this function"]
     fn foo(arr: Array) -> usize as bar;

     // Exports the function `foo` as `bar!` in the `Base` module.
     //
     // This syntax can be used to extend existing functions.
     fn foo(arr: Array) -> usize as Base.bar!;

     // Exports the struct `MyType` as `MyForeignType`. `MyType` must implement `OpaqueType`
     // or `ForeignType`.
     struct MyType as MyForeignType;

     // Exports `MyType::new` as `MyForeignType`, turning it into a constructor for that type.
     //
     // A Rust function is generated to call this method, so unlike free-standing functions
     // exported methods don't have to use the C ABI.
     in MyType fn new(arg0: TypedValue<u32>) -> TypedValueRet<MyType> as MyForeignType;

     // Exports `MyType::add` as the function `increment!`.
     //
     // Methods that take `self` in some way must return a `RustResultRet` because the
     // generated function tracks the borrow of `self` before calling the exported method. If
     // `self` is taken by value, it's cloned after being tracked.
     in MyType fn add(&mut self, incr: u32) -> RustResultRet<u32>  as increment!;

     // Exports the function `long_running_func`, the returned closure is executed on another
     // thread.
     //
     // After dispatching the closure to another thread, the generated Julia function waits for
     // the closure to return using an `AsyncCondition`. Because the closure is executed on
     // another thread you can't call Julia functions or allocate Julia data from it, but it is
     // possible to (mutably) access Julia data by tracking it.
     //
     // In order to be able to use tracked data from the closure,  `Unbound` managed types must
     // be used. Only `(Typed)ValueUnbound` and `(Typed)ArrayUnbound` exist,  they're aliases
     // for `(Typed)Value` and `(Typed)Array` with static lifetimes. The generated Julia
     // function guarantees all data passed as an argument lives at least until the closure has
     // finished, the tracked data must only be shared with that closure.
     async fn long_running_func(
         array: ArrayUnbound
     ) -> JlrsResult<impl AsyncCallback<i32>>;

     // Exports `MY_CONST` as the constant `MY_CONST`, its type must implement `IntoJulia`.
     // `MY_CONST` can be defined in Rust as either static or constant data, i.e. both
     // `static MY_CONST: u8 = 1` and `const MY_CONST: u8 = 1` can be exposed this way.
     const MY_CONST: u8;

     // Exports `MY_CONST` as the global `MY_GLOBAL`, its type must implement `IntoJulia`.
     // `MY_CONST` can be defined in Rust as either static or constant data, i.e. both
     // `static MY_CONST: u8 = 1` and `const MY_CONST: u8 = 1` can be exposed this way.
     static MY_CONST: u8 as MY_GLOBAL;
 }
 */