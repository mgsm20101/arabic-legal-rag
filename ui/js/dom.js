/*
 * DOM helpers, the seconds counter, and announcements. Text reaches the page
 * only through textContent and attribute values.
 */
import { ANNOUNCE_DELAY_MS, TEXT, TICK_MS, TOAST_MS } from "./copy.js";

export function byId(id) {
  const node = document.getElementById(id);
  if (node === null) throw new Error(`app.html has no #${id}`);
  return node;
}

export function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}

export function fromTemplate(id) {
  return byId(id).content.firstElementChild.cloneNode(true);
}

export function icon(name) {
  const svg = fromTemplate("tpl-icon");
  svg.querySelector("use").setAttribute("href", `#i-${name}`);
  return svg;
}

export function setText(node, text) {
  node.textContent = text;
  node.hidden = text === "";
}

export function setBusy(button, busy) {
  button.classList.toggle("is-busy", busy);
  const label = button.querySelector(".btn-label");
  if (label !== null && label.dataset.idle && label.dataset.busy) {
    label.textContent = busy ? label.dataset.busy : label.dataset.idle;
  }
}

/** Counts whole seconds into `node` and hands back a stop callback. */
export function startClock(node) {
  const started = Date.now();
  const timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    node.textContent = TEXT.elapsed(seconds);
    node.hidden = seconds < 1;
  }, TICK_MS);
  return () => clearInterval(timer);
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

export function reveal(node, block) {
  node.scrollIntoView({ block, behavior: reducedMotion.matches ? "auto" : "smooth" });
}

/**
 * Hands back announce(message, options): says the message through `region`, a
 * live status region, and also shows it as a toast unless `visible` is false.
 */
export function createAnnouncer(region) {
  let timer = 0;
  return (message, { tone = "info", visible = true } = {}) => {
    clearTimeout(timer);
    region.classList.remove("is-visible");
    region.textContent = "";
    // emptied first, so a repeated message is still a change that gets announced
    timer = setTimeout(() => {
      region.dataset.tone = tone;
      region.textContent = message;
      region.classList.toggle("is-visible", visible);
      timer = setTimeout(() => region.classList.remove("is-visible"), TOAST_MS);
    }, ANNOUNCE_DELAY_MS);
  };
}
