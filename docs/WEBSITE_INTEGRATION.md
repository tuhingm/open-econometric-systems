# Integrating OES into the academic website

The academic site is a static GitHub Pages site at `https://tuhingm.github.io/almamun/`. Keep the existing page structure and visual language. Add a navigation item for **Econometric Estimation Laboratory** and mount the lab in a dedicated page or section.

## 1. Add the client

Set the API URL to the deployed HTTPS service, then load the client:

```html
<script>window.OES_API_BASE = "https://YOUR-OES-SERVICE.example.com";</script>
<script src="/almamun/oes-frontend/oes-client.js" defer></script>
```

The placeholder URL must be replaced with the actual deployed service URL.

## 2. Check API availability

```javascript
const catalogue = await OES.models();
const runnable = catalogue.models.filter(model => model.status === "implemented");
```

Render planned models as unavailable/roadmap entries; never expose a Run button for a planned method.

## 3. Execute OLS

```javascript
const result = await OES.estimateOLS({
  data: {
    y: [1.2, 2.1, 2.8, 4.2, 5.1],
    x: [[1, 0.2], [1, 0.4], [1, 0.6], [1, 0.8], [1, 1.0]]
  },
  variable_names: ["const", "x1"],
  covariance: "HC1",
  confidence_level: 0.95
});
console.table(result.coefficients);
```

## 4. Production requirements

Use HTTPS, set exact CORS origins, add request limits and rate limiting, and never store uploaded datasets unless an explicit retention and deletion design is implemented. Do not put API keys or secrets in the static website JavaScript.
