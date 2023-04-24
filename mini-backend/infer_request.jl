module MyRustModule
using JlrsCore.Wrap

@wrapmodule("path/to/lib", :init_infer_request)

function __init__()
    @initjlrs
end
end