pub struct JbError {
    message_: String,
}

impl JbError {
    pub fn Message(self) -> String {
        return self.message_;
    }
}