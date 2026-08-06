class AppError(Exception):
    """Unexpected internal server error"""
    status_code = 500
    detail = "Unexpected internal server error"

    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        if detail is not None:
            self.detail = detail
        super().__init__(f"{code}: {self.detail}")


class BadRequestError(AppError):
    """Bad request user error"""
    status_code = 400
    detail = "Bad request user error"


class NotFoundError(AppError):
    """Not found user error"""
    status_code = 404
    detail = "Not found user error"
