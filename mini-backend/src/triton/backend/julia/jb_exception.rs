use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct JuliaBackendException {
    msg: String,
}

impl JuliaBackendException {
    pub fn new(msg: String) -> Self {
        return JuliaBackendException { msg: msg };
    }
}

impl fmt::Display for JuliaBackendException {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "exception")
    }
}

impl error::Error for JuliaBackendException {}
