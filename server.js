const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT_DIR = __dirname;
loadEnvFile();

const PORT = Number(process.env.PORT) || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const STATIC_ROUTES = {
  "/": "index.html",
  "/index.html": "index.html",
  "/chat": "chat.html",
  "/chat.html": "chat.html",
};

const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".ico": "image/x-icon",
};

const server = http.createServer(async (req, res) => {
  try {
    const requestUrl = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    const { pathname } = requestUrl;

    if (req.method === "GET" && pathname === "/api/health") {
      return sendJson(res, 200, {
        status: "ok",
        apiConfigured: Boolean(process.env.OPENAI_API_KEY),
        model: OPENAI_MODEL,
      });
    }

    if (req.method === "POST" && pathname === "/api/ask") {
      const body = await readJsonBody(req);
      const question = typeof body.question === "string" ? body.question.trim() : "";
      const topic = normalizeTopic(body.topic);

      if (!question) {
        return sendJson(res, 400, {
          error: "Please enter a health-related question before submitting.",
        });
      }

      if (question.length < 5) {
        return sendJson(res, 400, {
          error: "Please enter a little more detail so the AI can respond helpfully.",
        });
      }

      if (question.length > 500) {
        return sendJson(res, 400, {
          error: "Please keep the question under 500 characters.",
        });
      }

      if (!looksHealthRelated(question)) {
        return sendJson(res, 200, {
          ...outOfScopeResponse(),
          topic,
          model: OPENAI_MODEL,
          inScope: false,
        });
      }

      const result = await callOpenAI(question, topic);
      return sendJson(res, 200, {
        ...result,
        topic,
        model: OPENAI_MODEL,
      });
    }

    if (req.method === "GET" && STATIC_ROUTES[pathname]) {
      return serveFile(res, path.join(ROOT_DIR, STATIC_ROUTES[pathname]));
    }

    if (req.method === "GET" && pathname === "/favicon.ico") {
      res.writeHead(204);
      res.end();
      return;
    }

    sendJson(res, 404, { error: "Route not found." });
  } catch (error) {
    const statusCode = error.statusCode || 500;
    const message =
      statusCode >= 500
        ? error.publicMessage || "The server could not complete the request."
        : error.message;

    console.error("[server]", error);
    sendJson(res, statusCode, { error: message });
  }
});

server.listen(PORT, () => {
  console.log(`AIcura server running at http://localhost:${PORT}`);
});

function loadEnvFile() {
  const envPath = path.join(ROOT_DIR, ".env");

  if (!fs.existsSync(envPath)) {
    return;
  }

  const envContent = fs.readFileSync(envPath, "utf8");

  for (const rawLine of envContent.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (!line || line.startsWith("#")) {
      continue;
    }

    const separatorIndex = line.indexOf("=");

    if (separatorIndex === -1) {
      continue;
    }

    const key = line.slice(0, separatorIndex).trim();
    const value = line.slice(separatorIndex + 1).trim().replace(/^"(.*)"$/, "$1");

    if (key && !(key in process.env)) {
      process.env[key] = value;
    }
  }
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}

function serveFile(res, filePath) {
  fs.readFile(filePath, (error, fileBuffer) => {
    if (error) {
      sendJson(res, 404, { error: "File not found." });
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "Content-Type": CONTENT_TYPES[ext] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(fileBuffer);
  });
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let rawBody = "";

    req.on("data", (chunk) => {
      rawBody += chunk;

      if (rawBody.length > 1_000_000) {
        const error = new Error("Request body is too large.");
        error.statusCode = 413;
        reject(error);
        req.destroy();
      }
    });

    req.on("end", () => {
      if (!rawBody) {
        resolve({});
        return;
      }

      try {
        resolve(JSON.parse(rawBody));
      } catch (_error) {
        const error = new Error("Invalid JSON body.");
        error.statusCode = 400;
        reject(error);
      }
    });

    req.on("error", (error) => {
      reject(error);
    });
  });
}

function normalizeTopic(topic) {
  return topic === "eye-health" ? "eye-health" : "general-health";
}

async function callOpenAI(question, topic) {
  if (!process.env.OPENAI_API_KEY) {
    const error = new Error("OPENAI_API_KEY is missing. Add it to your .env file before calling the AI API.");
    error.statusCode = 500;
    error.publicMessage = error.message;
    throw error;
  }

  const systemPrompt = buildSystemPrompt(topic);
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      temperature: 0.3,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: question },
      ],
    }),
  });

  const responseBody = await response.json();

  if (!response.ok) {
    const error = new Error(responseBody.error?.message || "The AI API request failed.");
    error.statusCode = response.status === 429 ? 503 : 502;
    error.publicMessage = "The AI service is unavailable right now. Please try again.";
    throw error;
  }

  const messageContent = responseBody.choices?.[0]?.message?.content;

  if (!messageContent) {
    const error = new Error("The AI API returned an empty response.");
    error.statusCode = 502;
    error.publicMessage = "The AI service returned an empty response. Please try again.";
    throw error;
  }

  let parsedResult;

  try {
    parsedResult = JSON.parse(stripCodeFence(messageContent));
  } catch (_error) {
    if (!looksHealthRelated(question)) {
      return {
        ...outOfScopeResponse(),
        inScope: false,
      };
    }

    parsedResult = {
      answer: messageContent.trim(),
      explanation: fallbackExplanation(topic),
      disclaimer: fallbackDisclaimer(),
      urgency: "routine",
      in_scope: true,
    };
  }

  const inScope = normalizeInScope(parsedResult.in_scope);

  if (!inScope) {
    return {
      ...outOfScopeResponse(),
      inScope: false,
    };
  }

  return {
    answer: cleanText(parsedResult.answer) || "The AI service did not return an answer.",
    explanation: cleanText(parsedResult.explanation) || fallbackExplanation(topic),
    disclaimer: cleanText(parsedResult.disclaimer) || fallbackDisclaimer(),
    urgency: normalizeUrgency(parsedResult.urgency),
    inScope: true,
  };
}

function buildSystemPrompt(topic) {
  const scope =
    topic === "eye-health"
      ? "Prioritize eye and vision health context when relevant."
      : "Provide broad educational health information in plain language.";

  return [
    "You are AIcura, an educational healthcare information assistant.",
    scope,
    "You only answer health-related questions.",
    "Allowed topics include symptoms, wellness, preventive care, medications, general medical information, eye health, retinal scans, heart health, diabetes, and medical safety guidance.",
    "If a request is not about health, medical care, wellness, retinal scans, eye health, heart health, diabetes, or a closely related health topic, do not answer it.",
    'For out-of-scope requests, return JSON with "in_scope": false and tell the user that AIcura only answers health-related questions.',
    "Do not diagnose, prescribe treatment, confirm a condition, or present yourself as a medical professional.",
    "Never state or imply that the user definitely has a condition.",
    'Use careful phrasing such as "possible explanations include", "this can sometimes be associated with", or "people may experience this when".',
    "If symptoms suggest an emergency, give general safety guidance to seek urgent medical attention immediately.",
    "Keep the answer concise, practical, and understandable for non-experts.",
    'Return only valid JSON with the keys "in_scope", "answer", "explanation", "disclaimer", and "urgency".',
    'Use one of these urgency values exactly: "routine", "soon", "urgent". The urgency field is general care guidance only, not a diagnosis.',
    'The "answer" should be 2-4 sentences.',
    'The "explanation" should be 1-2 sentences about what factors matter.',
    'The "disclaimer" must clearly say the response is for educational purposes only, is not medical advice, and is not a diagnosis.',
  ].join(" ");
}

function stripCodeFence(text) {
  return text.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/\s*```$/, "").trim();
}

function cleanText(value) {
  return typeof value === "string" ? value.replace(/\s+/g, " ").trim() : "";
}

function normalizeUrgency(value) {
  return ["routine", "soon", "urgent"].includes(value) ? value : "routine";
}

function normalizeInScope(value) {
  if (typeof value === "string") {
    return value.toLowerCase() !== "false";
  }

  return value !== false;
}

function looksHealthRelated(question) {
  const normalizedQuestion = question.toLowerCase();
  const healthKeywords = [
    "health",
    "medical",
    "medicine",
    "doctor",
    "clinic",
    "hospital",
    "symptom",
    "symptoms",
    "pain",
    "fever",
    "cough",
    "cold",
    "flu",
    "headache",
    "migraine",
    "dizzy",
    "dizziness",
    "nausea",
    "vomit",
    "diarrhea",
    "constipation",
    "rash",
    "swelling",
    "infection",
    "injury",
    "wound",
    "fatigue",
    "tired",
    "sleep",
    "stress",
    "anxiety",
    "depression",
    "mental health",
    "therapy",
    "medication",
    "medicine",
    "pharmacy",
    "dose",
    "supplement",
    "vitamin",
    "diet",
    "nutrition",
    "exercise",
    "hydration",
    "allergy",
    "asthma",
    "breathing",
    "shortness of breath",
    "blood pressure",
    "cholesterol",
    "heart",
    "cardiac",
    "diabetes",
    "glucose",
    "insulin",
    "retina",
    "retinal",
    "eye",
    "vision",
    "blurry",
    "blurred",
    "dry eyes",
    "red eye",
    "fundus",
    "screening",
    "treatment",
    "wellness",
    "throat",
    "chest",
    "stomach",
    "abdomen",
    "skin",
    "back pain",
    "joint",
    "pregnancy",
  ];

  return healthKeywords.some((keyword) => normalizedQuestion.includes(keyword));
}

function outOfScopeResponse() {
  return {
    answer: "AIcura only answers health-related questions. Please ask about symptoms, wellness, eye or retinal health, heart health, diabetes, or other medical topics.",
    explanation: "Your message did not appear to be about a health-related topic, so no health guidance was generated.",
    disclaimer: fallbackDisclaimer(),
    urgency: "routine",
  };
}

function fallbackExplanation(topic) {
  if (topic === "eye-health") {
    return "This summary highlights common eye-health considerations related to your question and should be read as general educational information.";
  }

  return "This summary focuses on common health information related to your question and should be read as general educational guidance.";
}

function fallbackDisclaimer() {
  return "For educational purposes only. This AI-generated response is not medical advice, is not a diagnosis, and does not replace a licensed healthcare professional.";
}
