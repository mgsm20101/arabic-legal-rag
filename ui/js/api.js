/*
 * The one way the page talks to the server, and the checks every value from
 * the server passes before the page uses it.
 */
import { TEXT, retryAfterText } from "./copy.js";

export const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
export const wholeNumber = (value) => (Number.isInteger(value) && value >= 0 ? value : null);
export const textOf = (value, fallback) =>
  typeof value === "string" && value.trim() !== "" ? value.trim() : fallback;

export class ApiError extends Error {
  constructor(status, code, messageAr) {
    super(`${status} ${code}`);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.messageAr = messageAr;
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Same-origin paths only, the app header on every request, and every failure
 * turned into an ApiError that carries an Arabic sentence to show.
 */
export async function api(path, { method = "GET", json, form, signal } = {}) {
  const headers = { "X-LegalRAG": "1", Accept: "application/json" };
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form !== undefined) {
    body = form;
  }

  let response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body,
      signal,
      cache: "no-store",
      credentials: "same-origin",
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(0, "network", TEXT.networkError);
  }

  const data = await readJson(response);
  if (!response.ok) {
    const payload = isObject(data) ? data : {};
    const message = textOf(payload.message_ar, TEXT.genericError);
    const code = typeof payload.error === "string" ? payload.error : `http_${response.status}`;
    const wait = response.status === 429 ? ` ${retryAfterText(response.headers.get("Retry-After"))}` : "";
    throw new ApiError(response.status, code, message + wait);
  }
  if (!isObject(data)) throw new ApiError(response.status, "bad_response", TEXT.genericError);
  return data;
}

/** The Arabic sentence to show for a failure; an unexpected error is logged, never swallowed. */
export function messageOf(error) {
  if (error instanceof ApiError) return error.messageAr;
  console.error(error);
  return TEXT.genericError;
}
