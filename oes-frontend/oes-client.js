/* OES browser client. Configure OES_API_BASE before loading this file. */
(function (global) {
  "use strict";

  const API_BASE = (global.OES_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

  async function request(path, options = {}) {
    const response = await fetch(API_BASE + path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) }
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = payload.detail || payload.message || ("HTTP " + response.status);
      throw new Error(typeof message === "string" ? message : JSON.stringify(message));
    }
    return payload;
  }

  global.OES = {
    apiBase: API_BASE,
    async health() {
      return request("/health");
    },
    async models() {
      return request("/api/v1/models");
    },
    async estimateOLS(specification) {
      return request("/api/v1/estimate/ols", {
        method: "POST",
        body: JSON.stringify(specification)
      });
    }
  };
})(window);
