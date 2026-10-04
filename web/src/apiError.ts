/** A failed request, with the API's error code ("not_found", "offline", …) so screens can tell failures apart. */
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}
